from email.mime import image
from tkinter import Frame
from turtle import width
import cv2 
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.datasets import fetch_openml 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from PIL import Image 
import PIL.ImageOps 
import os, ssl, time


#fetchin the data

X,y=fetch_openml('mnist_784',version=1,return_X_y=True)
print(pd.Series(y).value_counts())
classes=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
n_classes=len(classes)

x_train,x_test,y_train,y_test=train_test_split(X,y ,random_state=9,train_size=7500,test_size=2500)
x_train_scalled=x_train/255.0
x_test_scalled=x_test/255.0

clf=LogisticRegression(solver='saga',multi_class='multinomial').fit(x_test_scalled,y_train)
y_pred=clf.predict(x_test_scalled)

print(accuracy=accuracy_score(y_test,y_pred))


cap=cv2.VideoCapture(0)
while(True):
    try:
       ret,frame=cap.read()
       gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
       height,width=gray.shape()
       upper_left = (int(width / 2 - 56), int(height / 2 - 56)) 
       bottom_right = (int(width / 2 + 56), int(height / 2 + 56))
       cv2.rectangle(gray,upper_left,bottom_right,(0,255,0),2)
       
       roi = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]
       im_pill=Image.fromarray(roi)
       imagebw=im_pill.convert('L')
       imagebw_resize=imagebw.resize((28,28),Image.ANTIALIAS)
       imagebw_resize_inverted=PIL.ImageOPS.invert(imagebw_resize)
       pixel_filter=20
       minimum_pixel=np.percentile(imagebw_resize_inverted,pixel_filter)
       imagebw_resize_inverted_scalled=np.clip(imagebw_resize_inverted-minimum_pixel,0,255)
       max_pixel=np.max(imagebw_resize_inverted)

       imagebw_resize_inverted_scalled=np.asarray(imagebw_resize_inverted_scalled)/max_pixel
       test_sam=np.array(imagebw_resize_inverted_scalled).reshape(1,784)
       test_pred=clf.predict(test_sam)
       print("predicted class is ",test_pred)

       cv2.imshow('frame',gray)
       if cv2.waitKey(1) & 0xFF==ord('q'):
        break
    except Exception as e:
        pass

cap.release()
cv2.releaseAllWindows()





