# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:34:08 2020

@author: shoun
"""

import cv2
import numpy as np

img = cv2.imread('hand.png')
kernel = np.ones((5,5),np.uint8)



#imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow('Gray image of dog 1',imgGray)
#cv2.waitKey(0)
#imgCanny = cv2.Canny(img,150,200)
#cv2.imshow('CAnny image of dog 1',imgCanny)
#cv2.waitKey(0)



imgBlur = cv2.GaussianBlur(img,(11,17),0)
cv2.imshow('Blur image of dog 1',imgBlur)
cv2.waitKey(0)


imghsv = cv2.cvtColor(imgBlur,cv2.COLOR_BGR2HSV)
cv2.imshow('hsv image of hand',imghsv)
cv2.waitKey(0)


mask2 = cv2.inRange(imghsv , np.array([2,0,0]), np.array([20,255,255]))
cv2.imshow('masked image of the hand',mask2)
cv2.waitKey(0)


#increasing the edge thickness
imgDialation = cv2.dilate(mask2,kernel,iterations=5)
cv2.imshow('dilation image of original hand',imgDialation)
cv2.waitKey(0)


#decreasing the edge thickness is called erosion
#decreasing image thickness
imgerosion = cv2.erode(imgDialation,kernel,iterations=5)
cv2.imshow('eroded image of hand',imgerosion)
cv2.waitKey(0)

imgBlur2 = cv2.GaussianBlur(imgerosion,(11,17),0)
cv2.imshow('Blur 2 image of hand',imgBlur2)
cv2.waitKey(0)

ret , thresh = cv2.threshold(imgBlur2,125,255,cv2.THRESH_BINARY)
cv2.imshow('fina filtereed output of the hand image',thresh)
cv2.waitKey(0)

