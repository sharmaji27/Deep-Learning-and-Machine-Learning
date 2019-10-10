import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df=pd.read_csv('movie_dataset.csv')

features=['genres','keywords','title','director','cast']

def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]

for feature in features:
    df[feature] = df[feature].fillna('')


def combine_features(row):
    return row['genres']+' '+row['keywords']+' '+row['title']+' '+row['director']+' '+row['cast']

df['combined_features']=df.apply(combine_features,axis=1)

cv=CountVectorizer()
count_matrix=cv.fit_transform(df['combined_features'])

cos_simi=cosine_similarity(count_matrix)

movie=input('ENTER YOUR MOVIE >>')

ind=get_index_from_title(movie)

similar_movies=list(enumerate(cos_simi[ind]))

sorted_similar_movies=sorted(similar_movies,key=lambda x:x[1],reverse=True)

i=0
for m in sorted_similar_movies:
    print(get_title_from_index(m[0]))
    i=i+1
    if i>50:
        break

input()

