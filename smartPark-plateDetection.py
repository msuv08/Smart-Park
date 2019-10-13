'''
Created on Mar 3, 2018

@author: Mihir Suvarna
'''

import cv2
import numpy as np

image = cv2.imread("/Users/mihir/Downloads/270721.jpg") #Locate image file and paste path here.

cv2.namedWindow("Original Image",cv2.WINDOW_NORMAL) #Displays the original image. 
cv2.imshow("Original Image",image)

imageGray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) #This makes the image gray for the next step.
noise_removal = cv2.bilateralFilter(imageGray,9,75,75) #After making this image gray, noise must be removed.

cv2.namedWindow("Gray Converted Image",cv2.WINDOW_NORMAL) #Display the grayed out image.
cv2.imshow("Gray Converted Image",imageGray)

cv2.namedWindow("Image w/o noise",cv2.WINDOW_NORMAL) #Displays the noiseless image.
cv2.imshow("Image w/o noise",noise_removal)

histoEQ = cv2.equalizeHist(noise_removal) #This equalizes the histogram.

cv2.namedWindow("After Histogram EQ",cv2.WINDOW_NORMAL) #Displays image after a pass of histogram equalization.
cv2.imshow("After Histogram EQ",histoEQ)

structCoord = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)) #Removes small irrelevant objects from the image.
morphImg = cv2.morphologyEx(histoEQ,cv2.MORPH_OPEN,structCoord,iterations=15) 

cv2.namedWindow("Morphological Opening (15)",cv2.WINDOW_NORMAL) #Displays the image after 15 passes of morphological opening.
cv2.imshow("Morphological Opening (15)",morphImg)

morphImg = cv2.subtract(histoEQ,morphImg) #Removes shadows and other uneven sections of images.

cv2.namedWindow("Subtraction Image", cv2.WINDOW_NORMAL) #Displays the image after a pass of image subtraction.
cv2.imshow("Subtraction Image", morphImg)

ret,threshImg = cv2.threshold(morphImg,0,255,cv2.THRESH_OTSU) #Thresholds the image.

cannyImage = cv2.Canny(threshImg,250,255) #Finds the edges of large objects still left in the image.

cv2.namedWindow("Thresholded Image",cv2.WINDOW_NORMAL) #Displays the thresholded image.
cv2.imshow("Thresholded Image",threshImg)

cv2.namedWindow("Image after applying Canny",cv2.WINDOW_NORMAL) #Displays the image after canny.
cv2.imshow("Image after applying Canny",cannyImage)
cannyImage = cv2.convertScaleAbs(cannyImage)

kernel = np.ones((3,3), np.uint8) #Dilation is done here to further process the image.

imgDilated = cv2.dilate(cannyImage,kernel,iterations=1) #Displays the dilated image.
cv2.namedWindow("Dilation", cv2.WINDOW_NORMAL)
cv2.imshow("Dilation", imgDilated)

contours, hierarchy = cv2.findContours(imgDilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours= sorted(contours, key = cv2.contourArea, reverse = True)[:10]

screenIn = None

for c in contours: #This section will contour the image.
    objPerimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.06 * objPerimeter, True) 
    if len(approx) == 4:
        screenIn = approx 
        break
final = cv2.drawContours(image, [screenIn], -1, (0, 255, 0), 3)

cv2.namedWindow("Image with Selected Contour",cv2.WINDOW_NORMAL) #Displays the contoured image.
cv2.imshow("Image with Selected Contour",final)
mask = np.zeros(imageGray.shape,np.uint8)

new_image = cv2.drawContours(mask,[screenIn],0,255,-1,) #Draws the contours of the image.
new_image = cv2.bitwise_and(image,image,mask=mask)

cv2.namedWindow("Final Image",cv2.WINDOW_NORMAL) #Displays the final image.
cv2.imshow("Final Image",new_image)

cy,cr,cb = cv2.split(cv2.cvtColor(new_image,cv2.COLOR_RGB2YCrCb)) #Enhances the image with minor adjustments.
cy = cv2.equalizeHist(cy)
enhancedImg = cv2.cvtColor(cv2.merge([cy,cr,cb]),cv2.COLOR_YCrCb2RGB)

cv2.namedWindow("Enhanced Number Plate",cv2.WINDOW_NORMAL) #Displays the enhanced version of the Final Image.
cv2.imshow("Enhanced Number Plate",enhancedImg)

cv2.waitKey() #Wait on key press by user; click any key to close all windows.
