import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from sklearn.svm import SVC
from sklearn import metrics
import os
import csv

df=pd.read_csv('training_dataset.csv')

y=df['0']
X=df.drop('0',axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,shuffle=True)


svc=SVC(kernel='linear',C=0.1,gamma=10)

svc.fit(X_train,y_train)

joblib.dump(svc,'model/svc_0_to_9')

print('model trained ......')

pred=svc.predict(X_test)
print(metrics.accuracy_score(y_test,pred)*100)