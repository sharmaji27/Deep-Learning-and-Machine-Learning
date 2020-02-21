# importing libararies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing data
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# scaling/normalizing the data
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X = sc.fit_transform(X)

#initializing self organizing map
from minisom import MiniSom
som = MiniSom(x=10,y=10,input_len=15,sigma=1.0,learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data = X , num_iteration = 100)

#visualizing the som
from pylab import bone,pcolor,colorbar,plot,show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o','s']
colors = ['r','g']

for i,x in enumerate(X):
    w = som.winner(x)
    plot(w[0]+0.5,
         w[1]+0.5,
         markers[y[i]],
         markeredgecolor=colors[y[i]],
         markersize=10,
         markeredgewidth=2,
         markerfacecolor='None'
         )
show()

#finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(1,1)],mappings[(1,2)]),axis=0)
frauds = sc.inverse_transform(frauds)



