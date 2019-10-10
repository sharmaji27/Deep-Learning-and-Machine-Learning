import joblib
from PIL import ImageGrab
import cv2
from tkinter import *
import tkinter
from tkinter import messagebox
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from sklearn.svm import SVC
from sklearn import metrics
import os




model=joblib.load('model/svc_0_to_9')


im = ImageGrab.grab(bbox=(10, 230, 260, 476))
im.save('temp/for prediction.png')
image=cv2.imread('temp/for prediction.png')
gim=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
resized_image=cv2.resize(gim,(28,28),interpolation=cv2.INTER_AREA)

P=[]
for i in range(28):
    for j in range(28):
        if resized_image[i,j]<100:
            k=0
        else:
            k=1
        P.append(k)

pred=model.predict([P])

print('my prediction is >> ',pred[0])

ans=input('IS THE PREDICTION RIGHT (Y/N) >> ')

if ans=='Y' :
    print('YEAHHHHH !!!')


elif ans=='N':
    l=input('WHAT WAS THE NUMBER YOU TRIED TO PREDICT ? >> ')
    X=[]
    X.append(l)
    img = ImageGrab.grab(bbox=(10, 230, 260, 476))
    img.save('temp//ima.png')
    ima=cv2.imread('temp//ima.png')
    ima=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
    ima=cv2.resize(ima,(28,28))
    for i in range(28):
        for j in range(28):
            if ima[i, j] > 100:
                k = 1
            else:
                k = 0
            X.append(k)
    with open('training_dataset.csv','a+',newline='') as cfa:
        cf=csv.writer(cfa)
        cf.writerow(X)
    print('ADDED SUCCESSFULLY !!')