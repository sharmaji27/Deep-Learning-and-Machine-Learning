import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn import metrics
import os
import csv
import cv2


all_image_no=os.listdir('training data')

for no in all_image_no:
    all_no_images=os.listdir('training data/'+str(no))
    for img in all_no_images:
        image=cv2.imread('training data/'+str(no)+'/'+str(img))
        image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image=cv2.resize(image,(28,28),interpolation=cv2.INTER_AREA)
        label = no
        X=[]
        X.append(label)
        for i in range(28):
            for j in range(28):
                if image[i,j]>100:
                    k=1
                else:
                    k=0
                X.append(k)

        with open('training_dataset.csv','a+',newline='') as tdw:
            cr=csv.writer(tdw)
            cr.writerow(X)