import cv2
from numpy import asarray,savez_compressed
import os
from PIL import Image
from mtcnn.mtcnn import MTCNN

def find_face(path, required_size=(160, 160)):
    org_image=cv2.imread(path)
    org_image = cv2.cvtColor(org_image,cv2.COLOR_BGR2RGB)
    org_image=asarray(org_image)
    detector = MTCNN()
    # detect faces in the image
    faces = detector.detect_faces(org_image)
    x,y,w,h = faces[0]['box']
    x,y=abs(x),abs(y)
    #cv2.rectangle(org_image,(x,y),(x+w,y+h),(172,5,233),2)
    face = org_image[y:y+h,x:x+w]
    face = Image.fromarray(face)
    face = face.resize(required_size)
    face_array = asarray(face)
    return face_array

def load_faces(path):
    faces = []
    for img in os.listdir(path):
        faces.append(find_face(path+'\\'+img))
    return faces


def load_dataset(path):
    X=[]
    y=[]
    for characters in os.listdir(path):
        if not os.path.isdir(path+'\\'+characters):
            continue
        faces=load_faces(path+'\\'+characters)
        labels=[characters for _ in range(len(faces))]
        print('>loaded {} examples for class: {}'.format(len(faces), characters))
        X.extend(faces)
        y.extend(labels)
    return asarray(X),asarray(y)



train_X , train_y = load_dataset('faces\\train')
print(train_X.shape,train_y.shape)
test_X , test_y = load_dataset('faces\\val')
savez_compressed('faces-dataset.npz',train_X,train_y,test_X,test_y)

