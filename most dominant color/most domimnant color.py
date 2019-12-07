import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# setting number of rows for resizing
rows =120

img = cv2.imread('pic.png')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
print('org image shape --> ',img.shape)

# resizing image according to the org image aspect ratio
r,c,ch = img.shape
resized_img = cv2.resize(img,(int((c/r)*rows),rows))
print('resized image shape --> ',resized_img.shape)

cv2.imshow('resized img',resized_img)
cv2.waitKey(0)

# resizing the resized image so that now in the resized array each row shows each pixel starting from top left
pixels = np.reshape(resized_img,(-1,3))

kmeans = KMeans(n_clusters=8)
kmeans.fit(pixels)

# getting the cluster centres
colors = np.array(kmeans.cluster_centers_,dtype='uint')

# counting the no of points around each cluster centres and calculating their percentages
color_count = np.unique(kmeans.labels_,return_counts=True)[0]
percentages_of_colors = (np.unique(kmeans.labels_,return_counts=True)[1]/pixels.shape[0])*100

zipped_colors_and_percentages = list(zip(percentages_of_colors,colors))

# sorting them according to their percentages
sorted_colors = sorted(zipped_colors_and_percentages,reverse=True)

i=0
block = np.ones((50,50,3),dtype='uint')

# creating the blocks
for per,color in sorted_colors:
    print(color)
    plt.subplot(1,8,i+1)
    block[:]=color
    plt.text(5,20,round(per,2))
    plt.imshow(block)
    plt.xticks([])
    plt.yticks([])
    i+=1
plt.show()

# creating the bar
bar = np.zeros((50,500,3),dtype='uint')
start_length=0

for i in range(len(sorted_colors)):
    per,color = sorted_colors[i]
    end_length = start_length + int((per*bar.shape[1])/100)
    if i==7:
        bar[:,start_length:] = color
    else:
        bar[:,start_length:end_length]=color
    print(start_length,end_length)
    start_length = end_length
    plt.xticks([])
    plt.yticks([])
plt.imshow(bar)
plt.show()
