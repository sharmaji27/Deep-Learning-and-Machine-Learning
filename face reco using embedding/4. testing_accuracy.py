from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

#load datasets
data = np.load('faces-dataset_embedded.npz')
train_X,train_y,test_X,test_y = data['arr_0'],data['arr_1'],data['arr_2'],data['arr_3']

# normalize input vectors
in_encoder = Normalizer(norm='l2')
train_X = in_encoder.transform(train_X)
test_X = in_encoder.transform(test_X)

# encoding labels
out_encoder=LabelEncoder()
train_y = out_encoder.fit_transform(train_y)
test_y  = out_encoder.transform(test_y)


#fit model
model = SVC(kernel='linear',probability=True)
model.fit(train_X,train_y)
joblib.dump(model,'SVM_model.sav')

# model = joblib.load('SVM_model.sav')

#make predictions
training_predictions = model.predict(train_X)
test_predictions = model.predict(test_X)

#acuuracy score
print('Training accuracy --> ',accuracy_score(train_y,training_predictions)*100)
print('Test accuracy --> ',accuracy_score(test_y,test_predictions)*100)
