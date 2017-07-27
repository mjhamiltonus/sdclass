# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

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

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I ....


Michael Hamilton


My pipeline started with that from the class. Extending the dashed
lines together was the tricky part.

I came across an idea on the internet of thresholding the slopes:

https://medium.com/@esmat.anis/robust-extrapolation-of-lines-in-video-using-linear-hough-transform-edd39d642ddf

That seemed to make sense, but, I wanted to try something more
adaptive, rather than manually setting more thresholds.  The website
also included maintaining history over the video. This makes great
sense, but, for this project I keep it simple with single image
processing only.

Using k-means, I found two clusters of slopes - one should be the
right line, and on the left. The larger group should be the dashed
lines, and the longer lines should be the solid line (note: two solid
lines are detected in many cases - the right and left side of the
solid line where the gradients are present in the image, as pulled out
by the Canny edge detection.)

So, here is how I extract the two different lines:

1. Compute slopes and lengths of all lines

2. Find the average and standard deviation of slopes

3. Remove lines with slopes that are > X standard deviations away
   (outlier rejection).  Based upon the assumption that most of the
   lines should be stripes given the masking that has been previously
   performed. X was initially set to 1.0, but, 1.5 was found to be
   better.

4. Perform k-means with 2 clusters. This identifies the right and
   left lines, and there are two dominant set of slopes.

In order to draw a single line on the left and right lanes, I modified
the draw_lines() function by

5. Use the two sets of clustered lines to create linear interpolations
   around the sets of lines.

6. Extrapolate both to the bottom of the image (although most likely,
   only the dashed line will need extrapolation).


If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline

The kmeans technique is based on the assumption that there are two
sets of lines.  If there is another set of lines, or edges, down the
middle of the lane for example, this technique will fail. The
additional line will pull the set of points used for interpolation off
of the lane.


### 3. Suggest possible improvements to your pipeline

This could be improved by using a simpler approach to detecting the
lane lines based on where they are expected to be - such as gating by
two specific sets of slopes. However, that technique would still
suffer during lane changes, as the strips would not be where
"expected" and would be missed.

