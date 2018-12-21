from flask import Flask,jsonify,request

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

#start of code from teste.py


import pandas as pd
import numpy as np
import Tkinter 
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
ratings = pd.read_csv('movielens.csv', sep=';', encoding='utf-8')


# ------------------------------------------------------------------------
# Content Based

from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')

titles = movies['title']
indices = pd.Series(movies.index, index=movies['title'])

dM = {}

def index2Name(lista):
    return(titles.iloc[lista].tolist())

def many2One(listas):
    d = {}
    for x in range(0,len(listas[0])):
            d[listas[0][x]]=x
    for l in listas[1:]:
        for x in range(0,len(l)):
            d[l[x]]+=x

    sorted_by_value = sorted(d.items(), key=lambda kv: kv[1])
    sorted_by_value = [i[0] for i in sorted_by_value]
    return(sorted_by_value)


# Function that get movie recommendations based on the cosine similarity score of movie features
def cbRecMatrix(feature):
    
    if (feature in ["actors", "country", "genre", "language", "writer"]):
        movies[feature] = movies[feature].str.split(', ')
    elif ( feature == "plot"):
        movies[feature] = movies[feature].str.split(' ')

    movies[feature] = movies[feature].fillna("").astype('str')
    matrix = tfidf_matrixGenre = tf.fit_transform(movies[feature])

    from sklearn.metrics.pairwise import linear_kernel
    cosine_sim = linear_kernel(matrix, matrix)
    return(cosine_sim)

def generateCBMatrix(features):
    for f in features:
        dM[f] = cbRecMatrix(f)
        print("GeneratedCBMatrix:" + f)

def cbRecommendations(title, features):

    idx = indices[title]
    mx = np.empty([len(titles),len(titles)])
    for f in features:
        mx[idx] += (dM[f[0]][idx] * f[1])

    sim_scores = list(enumerate(mx[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)  
    sim_scores = list(filter(lambda x: x[0] != idx, sim_scores))

    movie_indices = [i[0] for i in sim_scores]
    return(movie_indices)

#generateCBMatrix(['title', 'actors', 'country', 'genre', 'language', 'writer', 'plot', 'director', 'production'])

# ------------------------------------------------------------------------
# Collaborative Filtering Recommendation

fileModel = 'testeOutModel'

def startPredModel(user,ratings,fileOutput):

    reader = Reader ()

    data = Dataset.load_from_df(ratings[['userId', 'imdbId', 'rating']], reader)
    data.split(n_folds=5) # 5

    svd = SVD()
    evaluate(svd, data, measures=['RMSE', 'MAE'])

    trainset = data.build_full_trainset()
    svd.train(trainset)

    dump.dump(fileOutput,None,svd,1)

def cfRecommendations(user):
    ratings['userId'] = ratings['userId'].fillna(0)
    ratings['imdbId'] = ratings['imdbId'].fillna(0)

    ratings['rating'] = ratings['rating'].fillna(ratings['rating'].mean())

    pred,svd = dump.load(fileModel)

    vistos = ratings.loc[ratings['userId'] == user]
    vistosLista = vistos['imdbId'].tolist()

    list = []
    for mov in ratings.imdbId.unique():
        if (mov not in vistosLista):
            list.append((mov, svd.predict(user, mov, 3).est))

    list = sorted(list, key=lambda x: x[1], reverse=True)
    res = [ seq[0] for seq in list ]

    return(res)


# ------------------------------------------------------------------------
# Os melhores com base nos nossos utilizadores e outros sites

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
    if (site == 'meta'):
        result = ratings.sort_values(by=['metascore'], ascending=False)
    else:    
        result = ratings.sort_values(by=[site + '_rating'], ascending=False)

    return(list(result['imdbId']))












#end of code from teste.py









def listIDSToJSON(list):
  return jsonify(result = list)







app = Flask(__name__)

def hello():
    return "Hello World!"

@app.route("/")
def call():
  return hello()


@app.route("/test")
def testCall():
  res = jsonify(result=hello())
  return res


#Devido ao facto que podemos ter muitos dados nas features, vai ser enviado um JSON
@app.route("/contentBased",  methods=['POST'])
def callCbRecommendations():
    req_data = request.get_json()

    title = req_data['title']

    listRecomended = cbRecommendations(title,[])

    res = listIDSToJSON(listRecomended)

    return res


@app.route("/collaborativeBased/<int:user>")
def callCfRecommendations(user):
  startPredModel(user,ratings,fileModel)
  listRecomended = cfRecommendations(user)
  res = listIDSToJSON(listRecomended)
  return res


@app.route("/userBestRated")
def callUserBestRated():
  #get the list of recomendations
  listRecomended = userBestRated()


  # transform list into JSON to be later used on the other server
  res = listIDSToJSON(listRecomended)
  
  # return said JSON
  return res

@app.route("/userMostPopular")
def callUserMostPopular():
  listRecomended = userMostPopular()
  res = listIDSToJSON(listRecomended)
  return res


@app.route("/wsBestRated/<site>")
def callWsBestRated(site):
  listRecomended = wsBestRated(site)
  res = listIDSToJSON(listRecomended)
  return res
  
