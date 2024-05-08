import pandas as pd
import numpy as np
import os
from datetime import datetime

from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

pd.set_option('display.max_rows',50)
pd.set_option('display.max_columns', 50)

info = pd.read_csv('.//ml-100k/u.info' , sep=" ", header = None)
info.columns = ['Counts' , 'Type']

occupation = pd.read_csv('./ml-100k/u.occupation' , header = None)
occupation.columns = ['Occupations']

items = pd.read_csv('./ml-100k/u.item' , header = None , sep = "|" , encoding='latin-1')
items.columns = ['movie id' , 'movie title' , 'release date' , 'video release date' ,
              'IMDb URL' , 'unknown' , 'Action' , 'Adventure' , 'Animation' ,
              'Childrens' , 'Comedy' , 'Crime' , 'Documentary' , 'Drama' , 'Fantasy' ,
              'Film_Noir' , 'Horror' , 'Musical' , 'Mystery' , 'Romance' , 'Sci_Fi' ,
              'Thriller' , 'War' , 'Western']

data = pd.read_csv('./ml-100k/u.data', header= None , sep = '\t')
user = pd.read_csv('./ml-100k/u.user', header= None , sep = '|')
genre = pd.read_csv('./ml-100k/u.genre', header= None , sep = '|' )

genre.columns = ['Genre' , 'genre_id']
data.columns = ['user id' , 'movie id' , 'rating' , 'timestamp']
user.columns = ['user id' , 'age' , 'gender' , 'occupation' , 'zip code']

grouping_user = user

# Merging the columns with data table to better visualise
data = data.merge(user , on='user id')
data = data.merge(items , on='movie id')

# Data Cleaning for Model Based Recommandation System
def convert_time(x):
    return datetime.utcfromtimestamp(x).strftime('%d-%m-%Y')

def date_diff(date):
    d1 = date['release date'].split('-')[2]
    d2 = date['rating time'].split('-')[2]
    return abs(int(d2) - int(d1))

# data.drop(columns = ['movie title' , 'video release date' , 'IMDb URL'] , inplace = True)
data.dropna(subset = ['release date'] , inplace = True)

user_details = data.groupby('user id').size().reset_index()
user_details.columns = ['user id' , 'number of user ratings']
data = data.merge(user_details , on='user id')

movie_details = data.groupby('movie id').size().reset_index()
movie_details.columns = ['movie id' , 'number of movie ratings']
data = data.merge(movie_details , on='movie id')

user_details = data.groupby('user id')['rating'].agg('mean').reset_index()
user_details.columns = ['user id' , 'average of user ratings']
data = data.merge(user_details , on='user id')

movie_details = data.groupby('movie id')['rating'].agg('mean').reset_index()
movie_details.columns = ['movie id' , 'average of movie ratings']
data = data.merge(movie_details , on='movie id')


user_details = data.groupby('user id')['rating'].agg('std').reset_index()
user_details.columns = ['user id' , 'std of user ratings']
data = data.merge(user_details , on='user id')

movie_details = data.groupby('movie id')['rating'].agg('std').reset_index()
movie_details.columns = ['movie id' , 'std of movie ratings']
data = data.merge(movie_details , on='movie id')

data['age_group'] = data['age']//10
data['rating time'] = data.timestamp.apply(convert_time)
data['time difference'] = data[['release date' , 'rating time']].apply(date_diff, axis =1)

data['total rating'] = (data['number of user ratings']*data['average of user ratings'] + data['number of movie ratings']*data['average of movie ratings'])/(data['number of movie ratings']+data['number of user ratings'])
data['rating_new'] = data['rating'] - data['total rating']

del movie_details
del user_details

pivot_table_user = pd.pivot_table(data=data,values='rating_new',index='user id',columns='movie id')
pivot_table_user = pivot_table_user.fillna(0)
pivot_table_movie = pd.pivot_table(data=data,values='rating',index='user id',columns='movie id')
pivot_table_movie = pivot_table_movie.fillna(0)

user_based_similarity = 1 - pairwise_distances(pivot_table_user.values, metric="cosine")
movie_based_similarity = 1 - pairwise_distances(pivot_table_movie.T.values, metric="cosine")

user_based_similarity = pd.DataFrame(user_based_similarity)
user_based_similarity.columns = user_based_similarity.columns+1
user_based_similarity.index = user_based_similarity.index+1

movie_based_similarity = pd.DataFrame(movie_based_similarity)
movie_based_similarity.columns = movie_based_similarity.columns+1
movie_based_similarity.index = movie_based_similarity.index+1

def rec_movie(movie_id, num_movies=10):
    temp_table = pd.DataFrame(columns=items.columns)
    movies = movie_based_similarity[movie_id].sort_values(ascending=False).index.tolist()[:num_movies + 1]
    for mov in movies:
        temp_table = pd.concat([temp_table, items[items['movie id'] == mov]], ignore_index=True)
    return temp_table

print(rec_movie(500))