import pandas as pd
import numpy as np
import tkinter 
import matplotlib.pyplot as plt
import re

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD, evaluate, dump

import pprint

import warnings; warnings.simplefilter('ignore')


movies = pd.read_csv('filmes.csv', sep=';', encoding='utf-8')

# ------------------------------------------------------------------------
# Content Based

from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')

# Build a 1-dimensional array with movie titles
titles = movies['title']
indices = pd.Series(movies.index, index=movies['title'])

# Function that get movie recommendations based on the cosine similarity score of movie genres
def cbRecMatrix(title, feature):

    # Break up the big genre string into a string array and Convert genres to string value
    movies[feature] = movies[feature].str.split(', ')
    movies[feature] = movies[feature].fillna("").astype('str')

    matrix = tfidf_matrixGenre = tf.fit_transform(movies[feature])

    from sklearn.metrics.pairwise import linear_kernel
    cosine_sim = linear_kernel(matrix, matrix)
    return(cosine_sim)

def cbRecommendations(title, features, filters, nRes):

    mx = np.empty([len(titles),len(titles)])
    for f in features:
        mx += cbRecMatrix(title, f)
        
    idx = indices[title]
    sim_scores = list(enumerate(mx[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    for f in filters:
        sim_scores = list(filter(lambda x: movies[f][x[0]] == movies[f][idx], sim_scores))
  
    sim_scores = list(filter(lambda x: x[0] != idx, sim_scores))
    sim_scores = sim_scores[0:nRes]

    movie_indices = [i[0] for i in sim_scores]

    return titles.iloc[movie_indices].tolist()


# ------------------------------------------
# - Print Resultados
#print(
# cbRecommendations('Dial M for Murder', ['genre','actors'],[], 20)
#)
# "actors";"awards";"country";"director";"genre";"imdb_rating";"imdb_votes";"language";"metascore";"plot";"poster";"production";"ratings";"title";"writer";"year";"dvdYear";"releasedMonth";"releasedYear";"duration"
# "country";"director";"genre";"imdb_rating";"imdb_votes";"language";"metascore";"plot";"poster";"production";"ratings";"title";"writer";"year";"dvdYear";"releasedMonth";"releasedYear";"duration"


# ------------------------------------------------------------------------
# Collaborative Filtering Recommendation

fileModel = 'testeOutModel'

def startPredModel(user,ratings,fileOutput):

    reader = Reader ()

    data = Dataset.load_from_df(ratings[['userId', 'imdbId', 'rating']], reader)
    data.split(n_folds=2)

    svd = SVD()
    evaluate(svd, data, measures=['RMSE', 'MAE'])

    trainset = data.build_full_trainset()
    svd.train(trainset)


    dump.dump(fileOutput,None,svd,1)



def cfRecommendations(user):
    ratings = pd.read_csv('movielens.csv', sep=';', encoding='utf-8')

    # Fill NaN values in user_id and movie_id column with 0
    ratings['userId'] = ratings['userId'].fillna(0)
    ratings['imdbId'] = ratings['imdbId'].fillna(0)

    # Replace NaN values in rating column with average of all values
    ratings['rating'] = ratings['rating'].fillna(ratings['rating'].mean())

    # "userId";"rating";"imdbId"


    #inicializa o modelo e guarda para um ficheiro
    startPredModel(user,ratings,fileModel)


    # o load retorna um tuplo, mas neste caso o pred não tem nada com significado
    pred,svd = dump.load(fileModel)

    list = []
    for mov in ratings.imdbId.unique():
        list.append((mov, svd.predict(user, mov, 3).est))

    list = sorted(list, key=lambda x: x[1], reverse=True)


    return(list)

def cfRecommendationsExcludingList(user,listExcluded):
    listAux = cfRecommendations(user)
    listRec = list(filter(lambda x: x[0] not in listExcluded , listAux))
    return (
        listRec[0]
    )

def userBestRated():

    ratings = pd.read_csv('votesMovie.csv', sep=';', encoding='utf-8')
    result = ratings.sort_values(by=['av'], ascending=False)

    return(list(result['imdbId']))

def userMostPopular():

    ratings = pd.read_csv('votesMovie.csv', sep=';', encoding='utf-8')
    result = ratings.sort_values(by=['nov'], ascending=False)

    return(list(result['imdbId']))

def wsBestRated(site):

    ratings = pd.read_csv('filmes.csv', sep=';', encoding='utf-8')
    result = ratings.sort_values(by=[site+ '_rating'], ascending=False)

    return(list(result['imdbId']))

def wsMostPopular(site):

    ratings = pd.read_csv('filmes.csv', sep=';', encoding='utf-8')
    result = ratings.sort_values(by=[site+ '_votes'], ascending=False)

    return(list(result['imdbId']))

# http://nbviewer.jupyter.org/github/khanhnamle1994/movielens/blob/master/Deep_Learning_Model.ipynb
# https://www.kaggle.com/rounakbanik/movie-recommender-systems
# https://www.datacamp.com/community/tutorials/recommender-systems-python
# https://www.ethanrosenthal.com/2015/11/02/intro-to-collaborative-filtering/