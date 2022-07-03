
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# This file provides functionality for binarizing images/
# removing the background 


## CHANGE THIS 
filename = "/Users/Arina/Desktop/9/well_39/croppedImage39_9.png"


oneWell = cv.imread(filename)
#cv.waitKey(0)
# Covert to grayscale
imgray = cv.cvtColor(oneWell, cv.COLOR_BGR2GRAY)
#imgray = cv.blur(imgray, (3,3))

"""
# Find Canny edges
edged = cv.Canny(imgray, 30, 200)
# Finding Contours
# Use a copy of the image e.g. edged.copy()
# since findContours alters the image
contours, hierarchy = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
cv.imshow('Canny Edges After Contouring', edged)
cv.waitKey(0)

#ret, thresh = cv.threshold(imgray, 127, 255, 0)

print("Number of Contours found = " + str(len(contours)))
cv.drawContours(oneWell, contours, -1, (0, 255, 0), 3)

plt.imshow(oneWell)
# as opencv loads in BGR format by default, we want to show it in RGB.
plt.show()
#cv.imshow('Contours', oneWell)
#cv.waitKey(0)
"""

#ret, thresh = cv.threshold(imgray, 127, 255, 0)
#find the contours from the thresholded image
#contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)


# create background image
bg = cv.dilate(oneWell, np.ones((5,5), dtype=np.uint8))
bg = cv.GaussianBlur(bg, (5,5), 1)
src_no_bg = 255 - cv.absdiff(oneWell, bg)

# threshold
maxValue = 255
thresh = 235
retval, dst = cv.threshold(src_no_bg, thresh, maxValue, cv.THRESH_BINARY_INV)
plt.imshow(dst)

_, binary = cv.threshold(imgray, 225, 255, cv.THRESH_BINARY_INV)


#Calculate Maximum Width of each worms using Distance Transform
distance = cv.distanceTransform(binary, cv.DIST_L2,5)
print("gray", binary.shape)
minv,maxv,minp,maxp = cv.minMaxLoc(distance)
print(distance)
print("maxv2", maxp)# (half)width2, pos
print("Min", minv*2, minp)
draw = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)
cv.line(draw, (0, maxp[1]), (binary.shape[1], maxp[1]), (250,0,0),2, -1)
cv.line(draw, (0, minp[1]), (binary.shape[1], minp[1]), (250,0,0),2, -1)
plt.figure(figsize=(15, 15))
plt.imshow(draw)
plt.show()