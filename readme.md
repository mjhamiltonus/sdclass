# **Finding Lane Lines on the Road** 

### Michael Hamilton
### Term 1, SDCND
### July, 2017
---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline started with that from the class. Extending the dashed
lines together was the tricky part.

I came across an idea on the internet of thresholding the slopes:

https://medium.com/@esmat.anis/robust-extrapolation-of-lines-in-video-using-linear-hough-transform-edd39d642ddf

In an effort to make the technique more robust and adaptive, rather
than manually setting more thresholds, I tried to use K-Means to cluster
the slopes of the detected lines.

Using K-Means, Two clusters of slopes are found. This should
correspond to the right line and the left line, since these are
dominant signal in the images.

It was anticipated that the larger group should be the dashed lines,
and the longer lines should be the solid line (note: two solid lines
are detected in many cases - the right and left side of the solid line
where the gradients are present in the image, as pulled out by the
Canny edge detection). This did not prove to be the case universally,
and ultimately right and left were determined by positions of the
y-coordinate (right/left) of the line.

This is the outline of the final processing pipeline:

1. Compute slopes and lengths of all lines

2. Find the average and standard deviation of slopes

3. Remove lines with slopes that are > X standard deviations away
   (outlier rejection). This was based upon the assumption that most
   of the lines should be stripes given the masking that has been
   previously performed. X was initially set to 1.0, but, 1.5 was
   found to be better.

4. Perform K-Means with 2 clusters. This identifies the right and
   left lines, and there are two dominant set of slopes.

In order to draw a single line on the left and right lanes, I modified
the draw_lines() function by

5. Use the two sets of clustered lines to create linear interpolations
   around the sets of lines.

6. Determine which line was right vs. left by looking at the mean of
   the y-coordinate.

6. Extrapolate both lines to the bottom of the image (although most
   likely, only the dashed line will need extrapolation), and to the
   y-coordinate at the top of the mask.


### 2. Identify potential shortcomings with your current pipeline

The K-Means technique is based on the assumption that there are two
sets of lines.  If there is another set of lines, or edges, down the
middle of the lane for example, this technique will fail. The
additional line will pull the set of points used for interpolation off
of the lane.

The approach clearly fails in the challenge case, where there are many
horizontal lines inside of the mask.

### 3. Suggest possible improvements to your pipeline

This could be improved by using a simpler approach to detecting the
lane lines based on where they are expected to be - such as gating by
two specific sets of slopes. However, that technique would still
suffer during lane changes, as the strips would not be where
"expected" and would be missed.

The website mentioned above also included maintaining history over the
video. This makes great sense, but, for this project the technique was
kept simple with single image processing only. Smoothing over frames
would likely have solved the issues present in the baseline
cases. Something different would be required in the challenge case.



