

# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# 
# ## Project: **Finding Lane Lines on the Road** 
# ***
# In this project, you will use the tools you learned about in the lesson to identify lane lines on the road.  You can develop your pipeline on a series of individual images, and later apply the result to a video stream (really just a series of images). Check out the video clip "raw-lines-example.mp4" (also contained in this repository) to see what the output should look like after using the helper functions below. 
# 
# Once you have a result that looks roughly like "raw-lines-example.mp4", you'll need to get creative and try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines.  You can see an example of the result you're going for in the video "P1_example.mp4".  Ultimately, you would like to draw just one line for the left side of the lane, and one for the right.
# 
# In addition to implementing code, there is a brief writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) that can be used to guide the writing process. Completing both the code in the Ipython notebook and the writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/322/view) for this project.
# 
# ---
# Let's have a look at our first image called 'test_images/solidWhiteRight.jpg'.  Run the 2 cells below (hit Shift-Enter or the "play" button above) to display the image.
# 
# **Note: If, at any point, you encounter frozen display windows or other confounding issues, you can always start again with a clean slate by going to the "Kernel" menu above and selecting "Restart & Clear Output".**
# 
# ---

# **The tools you have are color selection, region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Tranform line detection.  You  are also free to explore and try other techniques that were not presented in the lesson.  Your goal is piece together a pipeline to detect the line segments in the image, then average/extrapolate them and draw them onto the image for display (as below).  Once you have a working pipeline, try it out on the video stream below.**
# 
# ---
# 

# ## Import Packages

# In[1]:


#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
# get_ipython().magic('matplotlib inline')


# ## Read in an Image

# In[2]:


#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
# plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')


# ## Ideas for Lane Detection Pipeline

# **Some OpenCV functions (beyond those introduced in the lesson) that might be useful for this project are:**
# 
# `cv2.inRange()` for color selection  
# `cv2.fillPoly()` for regions selection  
# `cv2.line()` to draw lines on an image given endpoints  
# `cv2.addWeighted()` to coadd / overlay two images
# `cv2.cvtColor()` to grayscale or change color
# `cv2.imwrite()` to output images to file  
# `cv2.bitwise_and()` to apply a mask to an image
# 
# **Check out the OpenCV documentation to learn about these and discover even more awesome functionality!**

# ## Helper Functions

# Below are some helper functions to help get you started. They should look familiar from the lesson!

# In[3]:


import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


 
def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h, 4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf


# ## Test Images
# 
# Build your pipeline to work on the images in the directory "test_images"  
# **You should make sure your pipeline works well on these images before you try the videos.**

# In[4]:


import os
os.listdir("test_images/")


# ## Build a Lane Finding Pipeline
# 
# 

# Build the pipeline and run your solution on all test_images. Make copies into the `test_images_output` directory, and you can use the images in your writeup report.
# 
# Try tuning the various parameters, especially the low and high Canny thresholds as well as the Hough lines parameters.

# In[5]:

procImages = os.listdir("test_images/")
#for imName in procImages:
for ii in range(len(procImages)):
#    if ii != 3:
#      continue
    imName = procImages[ii]
    image = mpimg.imread('test_images/' + imName)
    # fig = plt.figure()
    # plt.imshow(image)
    # plt.show()
    
    # Process the image here
    grayIm = grayscale(image)

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_grayIm = cv2.GaussianBlur(grayIm,(kernel_size, kernel_size),0)

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_grayIm, low_threshold, high_threshold)

    # Next we'll create a masked edges image using cv2.fillPoly()
    mask = np.zeros_like(edges)   
    ignore_mask_color = 255   

    # This time we are defining a four sided polygon to mask.
    # TODO: Unfortunately I have y and x reversed nomenclature. Fix.
    imshape = image.shape
    # vertices = np.array([[(10,imshape[0]),(0,0), (imshape[1], 0), (imshape[1],imshape[0])]], dtype=np.int32
    xMax = imshape[0];       # bottom
    xMin = imshape[0]*0.60  # top
    yMin = imshape[1]*0.05
    yMaxL = imshape[1]*0.46
    yMaxR = imshape[1]*0.53
    yMax  = imshape[1]
    vertices = np.array([[(yMin,xMax),
                          (yMaxL,xMin),
                          (yMaxR,xMin), 
                          (yMax,xMax)]], 
                        dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 3     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 25 # minimum number of pixels making up a line
    max_line_gap    = 15 # maximum gap in pixels between connectable line segments
    line_image = np.copy(image)*0 # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = [];
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)

    # Iterate over the output "lines" and draw lines on a blank image
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

    # Should be two "clusters" of slopes: right and left line. 
    # TODO: line should be an object with these variables, instead of an array

    # Find the slopes and lengths of all lines
    nLines = lines.shape[0]
    mySlopes = np.zeros((nLines,1))
    myLengths = np.zeros((nLines,1))
    il=0  # index of line
    for line in lines:
        for x1,y1,x2,y2 in line:
            mySlopes[il] = ( (x2-x1)/(y2-y1) )
        il = il + 1
    # Find the mean and std of slope
    alpha = 1.5 # tuning parameter - number of std.devs. away allows
    lowSlope = np.mean(mySlopes) - alpha * np.std(mySlopes)
    highSlope= np.mean(mySlopes) + alpha * np.std(mySlopes)
    goodSlopes = np.zeros((nLines,1))
    goodLines = np.zeros(lines.shape, dtype=lines.dtype)
    goodLengths = np.zeros((nLines,1)) # same size
    iiGood = 0  # I use iiSomething for indices
    iiAll = 0
    for slope0 in mySlopes:
        if lowSlope < slope0 < highSlope:
            goodSlopes[iiGood] = slope0
            for x1,y1,x2,y2 in lines[iiAll]:
                goodLengths[iiGood] = ( np.sqrt( (x2-x1)**2+(y2-y1)**2 ) )
            goodLines[iiGood] = lines[iiAll]
            iiGood = iiGood + 1
        iiAll = iiAll + 1  
    # cut off the extra uninitialized - keep all in sync
    goodLines = goodLines[0:iiGood-1,:,:]
    goodSlopes = goodSlopes[0:iiGood-1]
    goodLengths = goodLengths[0:iiGood-1]

    # Find the two slopes
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LinearRegression
    kmeans = KMeans(n_clusters=2).fit(goodSlopes)

    goodLineImage = np.copy(image)*0 # blank for right/left image
    iiGood = 0
    xTrain0 = np.ndarray((1,0))
    yTrain0 = np.ndarray((1,0))
    xTrain1 = np.ndarray((1,0))
    yTrain1 = np.ndarray((1,0))
    for line in goodLines:
        for x1,y1,x2,y2 in line:
            if kmeans.labels_[iiGood] == 0:
#                cv2.line(goodLineImage,(x1,y1),(x2,y2),(255,0,0),10)
                xTrain0 = np.append(xTrain0, (x1, x2))
                yTrain0 = np.append(yTrain0, (y1, y2))
            else:
#                cv2.line(goodLineImage,(x1,y1),(x2,y2),(0,255,0),10)
                xTrain1 = np.append(xTrain1, (x1, x2))
                yTrain1 = np.append(yTrain1, (y1, y2))
            iiGood = iiGood + 1
            
            
    line0 = LinearRegression()
    line0.fit(xTrain0.reshape((xTrain0.size,1)), yTrain0.reshape((yTrain0.size,1)))
    line1 = LinearRegression()
    line1.fit(xTrain1.reshape((xTrain1.size,1)), yTrain1.reshape((yTrain1.size,1)))
    
    # Now I have to get right/left correct for the right model
    if(np.mean(xTrain1) > np.mean(xTrain0)):
        cv2.line(goodLineImage, 
                 (np.int_(0), np.int_(line0.predict(0))), 
                 (np.int_(yMaxL), np.int_(line0.predict(yMaxL))), 
                 (255,0,0), 10)
        cv2.line(goodLineImage, 
                 (np.int_(yMaxR), np.int_(line1.predict(yMaxR))), 
                 (np.int_(yMax), np.int_(line1.predict(yMax))), 
                 (255,0,0), 10)
    else:
        cv2.line(goodLineImage, 
                 (np.int_(0), np.int_(line1.predict(0))), 
                 (np.int_(yMaxL), np.int_(line1.predict(yMaxL))), 
                 (255,0,0), 10)
        cv2.line(goodLineImage, 
                 (np.int_(yMaxR), np.int_(line0.predict(yMaxR))), 
                 (np.int_(yMax), np.int_(line0.predict(yMax))), 
                 (255,0,0), 10)
    lines_image = cv2.addWeighted(image, 0.8, goodLineImage, 1, 0)
    plt.imshow(lines_image)
    plt.show()
    


            # input("press enter to continue")

    # Least Squares fit the two lines
    # pull the points based on class
    
    # interpolate them.
    # use the function created to get from ymax(bottom) to ymin(top)
    

    
            
    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges)) 

    # Draw the lines on the edge image
    lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0) 
    lines_image = cv2.addWeighted(image, 0.8, line_image, 1, 0) 
    lines_image = cv2.addWeighted(image, 0.8, goodLineImage, 1, 0) 
    fig = plt.figure()
    plt.imshow(lines_image)
    # overplot the selected region
    plt.plot( vertices[0,:,0], vertices[0,:,1], 'b--', lw=4)
    plt.show()

    # save image created to a buffer
    myImage = fig2data( fig )
    
#    input("press enter to continue")

    # write image to file
    cv2.imwrite('test_images_output/' + imName,  lines_edges);

print( 'all done here' );


    
