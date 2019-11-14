import numpy as np
from keras.models import load_model


def get_embeddings(model,face): # It will take face as (160,160,3)
    face = face.astype('float32')
    fm,fs = face.mean(),face.std()
    face = (face-fm)/fs
    samples = np.expand_dims(face,axis=0)
    y_pred = model.predict(samples)
    # print('predicted y',y_pred[0])
    return y_pred[0]

data = np.load('faces-dataset.npz')
train_X , train_y , test_X , test_y = data['arr_0'],data['arr_1'],data['arr_2'],data['arr_3']
print('Loaded data : ',train_X.shape,train_y.shape,test_X.shape,test_y.shape)

model = load_model('facenet_keras.h5')
print('Loaded model !!!')

new_train_x=[]
for face_pixels in train_X:
    new_train_x.append(get_embeddings(model,face_pixels))

new_test_x=[]
for face_pixels in test_X:
    new_test_x.append(get_embeddings(model,face_pixels))

new_test_x  = np.asarray(new_test_x)
new_train_x = np.asarray(new_train_x)

np.savez_compressed('faces-dataset_embedded.npz',new_train_x,train_y,new_test_x,test_y)

print('embedding done !!!!')