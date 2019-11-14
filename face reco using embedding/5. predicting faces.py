import cv2
from PIL import Image
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import Normalizer
from sklearn.externals import joblib
import os

#for converting ids to names
def id2name(id):
    x = os.listdir('faces\\train\\')
    return x[id]

ML_model = joblib.load('SVM_model.sav')
print('Loaded ML Model')

model = load_model('facenet_keras.h5')
print('Loaded embedding model !!!')

def get_embeddings(model,face):
    face = face.astype('float32')
    fm,fs = face.mean(),face.std()
    face = (face-fm)/fs
    samples = np.expand_dims(face,axis=0)
    print('ss',samples.shape)
    if samples.shape==(1,160,160,3):
        y_pred = model.predict(samples)
        return y_pred[0]
    else:
        return None

def get_predictions(face_array):
    if get_embeddings(model,face_array)is not None:
        face_emb = (get_embeddings(model,face_array))
        in_encoder = Normalizer(norm='l2')
        face_emb = in_encoder.transform(face_emb.reshape(1,-1))
        predicted_person = ML_model.predict(face_emb)
        return predicted_person
    else:
        return 'unknown'


def detect_face(frame):
    faces=face_classifier.detectMultiScale(frame,1.3,3)
    if faces==():
        return frame
    for x,y,w,h in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(172,42,251),2)
        face = frame[y:y+h,x:x+w]
        face = Image.fromarray(face)
        face= face.resize((160,160))
        face_array = np.asarray(face)   # we have made a face array of size (160,160,3)                                                         (B,G,R)
        cv2.putText(frame,text=str(id2name(int(get_predictions(face_array)))),org=(x,y-15),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(106,40,243),thickness=2)
    return frame

face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0)

while 1:
    ret,frame=cap.read()
    if ret==True:
        cv2.imshow('predictions',detect_face(frame))
        if cv2.waitKey(1)==27:
            break
cap.release()
cv2.destroyAllWindows()