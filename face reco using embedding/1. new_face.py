import os
import cv2

name=input('enter your name ---> ')
try:
    os.makedirs('faces\\train\\'+name)
    os.makedirs('faces\\val\\'+name)
except:
    pass
cap=cv2.VideoCapture(0)

i=0

while i<200:
    ret,frame = cap.read()
    if ret==True:
        cv2.imshow('taking your pictures',frame)
        #for training
        if i%5==0 and i<=150 and i!=0:
            cv2.imwrite('faces\\train\\'+name+'\\'+str(i)+'.jpg',frame)
        #for testing
        elif i % 5 == 0 and i > 150:
            cv2.imwrite('faces\\val\\'+name+'\\'+str(i) + '.jpg', frame)

        i=i+1
        print(i)
        if cv2.waitKey(1)==13:  #Enter Key
            break

cap.release()
cv2.destroyAllWindows()

