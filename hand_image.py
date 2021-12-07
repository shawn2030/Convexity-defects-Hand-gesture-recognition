# -*- coding: utf-8 -*-
"""
Created on Thu May  7 12:14:57 2020

@author: shoun
"""

import cv2
import numpy as np

# read the image first
hand = cv2.imread('hand.png',0)

#convert the image into a threshold image which is a binary image 
ret , thresh = cv2.threshold(hand,70,255,cv2.THRESH_BINARY)

#make a copy of threshold image
thresh_copy = thresh.copy()

#FIND THE CONTOURS OF THE THRESHOLD IMAGE
#find contours basically find the outer similar pixeels(in this case gray scale pixels) and chains/connect those pixels
#this  gives the outerlines or edges in an image
contours, hierarchy = cv2.findContours(thresh_copy,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


#tight fitting the points that lies in the interior so called convex points
hull =[cv2.convexHull(c) for c in contours]

#draw the final optimised contours
final = cv2.drawContours(hand,hull,-1,(255,0,0))


cv2.imshow('Original Hand',hand)
cv2.imshow('Threshold image',thresh)
cv2.imshow('convex hull image',final)

cv2.waitKey(0)
cv2.destroyAllWindows()
