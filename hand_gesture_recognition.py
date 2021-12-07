# -*- coding: utf-8 -*-
"""
Created on Thu May  7 17:32:48 2020

@author: shoun
"""

import numpy as np
import cv2
import math

#open camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    
    #capture frames from the camera
    ret , frame = cap.read()
    
    #get hand data from the rectangle sub window
    cv2.rectangle(frame,(100,100),(300,300),(0,255,0),0)
    crop_image = frame[100:300 , 100:300]
    
    #apply filtering with gaussian blur
    blur = cv2.GaussianBlur(crop_image,(7,7),0)
    
    ##change color space from BGR to HSV
    hsv = cv2.cvtColor(blur , cv2.COLOR_BGR2HSV)
    
    #create binary image with where white whill be skin color and rest is black
    mask2 = cv2.inRange(hsv , np.array([0,0,0]), np.array([20,255,255]))
    
    #kernel for morphological transformation
    kernel = np.ones((5,5))
    
    #apply morphological transformation "to filter out the backgroiund noise"
    dilation = cv2.dilate(mask2 , kernel , iterations = 1) 
    erosion = cv2.erode(dilation, kernel, iterations = 1)
    
    #apply Gaussian blur to filter it again
    filtered_img  = cv2.GaussianBlur(erosion,(3,3),0)
    ret , thresh = cv2.threshold(filtered_img,100,255,cv2.THRESH_BINARY)
    
    #show threshold image
    cv2.imshow('Thresholded output of the image',thresh)
    
    #find contours
    contours , hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    try :
        
        #find the contour with maximum area using area function in a lambda function
        contour = max(contours,key = lambda x:cv2.contourArea(x))
        
        #create a bounding rectangle around the contour
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(crop_image,(x,y),(x+w,y+h),(0,0,255),0)
        
        ##tight fitting the points that lies in the interior so called convex points
        hull =cv2.convexHull(contour)
        
        #draw contours
        drawing = np.zeros(crop_image.shape,np.uint8)
        cv2.drawContours(drawing,[contour],-1,(0,255,0),0)
        cv2.drawContours(drawing,[hull],-1,(0,0,255),0)
        
        
        #Finding the convexity defects
        hull = cv2.convexHull(contour,returnPoints=False)
        defects = cv2.convexityDefects(contour,hull)
        
        count_defects = 0
        
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            
            #a,b,c are the sides of a triangle that are formed by the points we get from strat,end,far
            #a is side opposite to far point and thus to find the angle at a we need inverse cosine function
            a = math.sqrt((end[0] - start[0])**2 + (end[1]-start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1]-start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1]-far[1])**2)
            
            #this is the inverse cosine function to find the angle
            angle = (math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 180) / 3.14
            
            
            #
            if angle <= 90:
                count_defects = count_defects + 1
                cv2.circle(crop_image,far,1,[0,0,255],-1)
                
            cv2.line(crop_image , start,end,[0,255,0],2)    
         
            
          #print the number of fingers on the live video
          
        if count_defects == 0:
              cv2.putText(frame,'ONE',(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),2)
        elif count_defects == 1:
              cv2.putText(frame,'TWO',(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),2)
        elif count_defects == 2:
              cv2.putText(frame,'THREE',(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),2)
        elif count_defects == 3:
              cv2.putText(frame,'FOUR',(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),2)
        elif count_defects == 4:
              cv2.putText(frame,'FIVE',(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),2)
        else:
            pass
    
    
    except:
            pass
     
        
    #show the required images we get from the live video
    cv2.imshow('Gesture',frame)

    #getting the joint form of images of contours and coprred image
    all_image = np.hstack((drawing,crop_image))
    cv2.imshow('contours',all_image)

    #close the camera if 's' is pressed 
    if cv2.waitKey(1) == ord('s'):
        break
    
    
cap.release()
cv2.destroyAllWindows()    