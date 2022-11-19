# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 14:34:00 2022

@author: USER
"""

# Importing necessary labraries
import numpy as np
import cv2

# classifiying Face 
face_classifier =cv2.CascaddeClassifier('F:\Datascience1\June\notes\OPEN - CV  FACE & EYE FROM VIDEO\Haarcascades\haarcascade_frontalface_default.xml')

#Loading the image then Coverting it to grayscale
image = cv2.imread('C:\Users\USER\Desktop\Documents for job\passphoto.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(gray,1.3,5)


#when no face Detected, face_classifier returns an empty tuple
if faces is ():
    print("no faces found")
    
    
#Creating rectangle 
# over each faces
for (x,y,w,h) in faces:
    cv2.rectagle(image, (x,y), (x+w,y+h),(127,0,255),2)
    cv2.imshow("face detected",image)
    cv2.waitKey(0)
    
#closing the all windows & stopping kernel
cv2.destroyALLWindows()