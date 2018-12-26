from flask import Flask,jsonify,request

#import sys
#from importlib import reload
#reload(sys)
#sys.setdefaultencoding("utf-8")

#start of code from teste.py


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

import pickle

import pprint
import warnings; warnings.simplefilter('ignore')

# ------------------------------------------------------------------------

movies = pd.read_csv('filmes.csv', sep=';', encoding='utf-8')
ratings = pd.read_csv('movielens.csv', sep=';', encoding='utf-8')

regex = re.compile('[^a-zA-Z]')
stemmer = SnowballStemmer('english')

from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')

# Build a 1-dimensional array with movie titles
titles = movies['imdb_id']
index = pd.Series(movies.index, index=movies['imdb_id'])
indices = pd.Series(movies.index, index=movies['title'])
# print(index)

cb = ['title', 'actors', 'country', 'genre', 'language', 'writer', 'plot', 'director', 'production']
dM = {} # Dicionario de Content Based

# Fill NaN values in user_id and movie_id column with 0
ratings['userId'] = ratings['userId'].fillna(0)
ratings['imdbId'] = ratings['imdbId'].fillna(0)

# Replace NaN values in rating column with average of all values
ratings['rating'] = ratings['rating'].fillna(ratings['rating'].mean())

# ------------------------------------------------------------------------
# Treatment

def index2imdbId(lista):
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

def utilizador2Vistos(user):
    vistos = ratings.loc[ratings['userId'] == user]
    return(vistos['imdbId'].tolist())

def imdb2index(imdb):
    return([index[imdb]][0])

# ------------------------------------------------------------------------
# Content Based

# Function that get movie recommendations based on the cosine similarity score of movie features
def cbRecMatrix(feature):
    
    # Break up the big genre string into a string array and Convert genres to string value
    if (feature in ["actors", "country", "genre", "language", "writer"]):
        movies[feature] = movies[feature].str.split(', ')
    elif ( feature == "plot"):
        
        movies[feature] = movies[feature].str.split(' ')
        for x in range(0, len(movies[feature])):
            # movies[feature][x] = [stemmer.stem(y) for y in movies[feature][x]]
            pass

    movies[feature] = movies[feature].fillna("").astype('str')
    matrix = tfidf_matrixGenre = tf.fit_transform(movies[feature])

    from sklearn.metrics.pairwise import linear_kernel
    cosine_sim = linear_kernel(matrix, matrix)
    return(cosine_sim)

def cbRecFromTitle(title, features):
    idx = indices[title]
    return (cbRecFromId(idx,features))

def cbRecFromImdb(title, features):
    idx = index[title]
    
    mx = np.empty([len(titles),len(titles)])
    for f in features:
        mx[0] += (dM[f[0]][idx] * f[1])

    return(mx[0])


def cbRecFromId(idx, features):

    mx = np.empty([len(titles),len(titles)])
    for f in features:
        mx[0] += (dM[f[0]][idx] * f[1])

    sim_scores = list(enumerate(mx[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)  
    sim_scores = list(filter(lambda x: x[0] != idx, sim_scores))

    movie_indices = [i[0] for i in sim_scores]
    return(index2imdbId(movie_indices))

def cbRecFromUser(user, features):

    lista = utilizador2Vistos(user)
    mx = np.empty([len(titles),len(titles)])
    for x in range(len(lista)):
        mx[0] += cbRecFromImdb(lista[x], features)

    sim_scores = list(enumerate(mx[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)  
    sim_scores = list(filter(lambda x: x[0] not in lista, sim_scores))

    movie_indices = [i[0] for i in sim_scores]
    return(index2imdbId(movie_indices))


# ------------------------------------------

def generateCBMatrix():
    for key in cb:
        print("Generating CBMatrix: " + key)
        dM[key] = cbRecMatrix(key)
    print("Geration Complete")

def saveCBMatrix():
    for key in dM:
        print("Saving CBMatrix: " + key)
        np.save('cb/' + key, dM[key])
    print("Save Complete")

def loadCBMatrix():
    for key in cb:
        print("Loading CBMatrix: " + key)
        dM[key] = np.load('./cb/' + key + '.npy') 
    print("Load Complete")


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
    # ratings = pd.read_csv('movielens.csv', sep=';', encoding='utf-8')

    # "userId";"rating";"imdbId"

    pred,svd = dump.load(str(fileModel),encoding='latin1')
    vistosLista = utilizador2Vistos(user)

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
    elif(site=='imdb'):    
        result = ratings.sort_values(by=['imdb_rating'], ascending=False)
    elif(site=='IMD'):    
        result = ratings.sort_values(by=['Internet Movie Database'], ascending=False)
    elif(site=='rotten'):    
        result = ratings.sort_values(by=['Rotten tomatoes'], ascending=False)

    return(list(result['imdb_id']))

# ------------------------------------------------------------------------
# Hibrido

def hibRecomend(user):
    pass    





#end of code from teste.py




#inicializing dM
#gera os ficheiros caso não existam

#garbage collector
#existe apenas por causa do meu computador ter memória muito limitada
#import gc
#
#import os
#
#for key in cb:
#    path = './cb/' + key + ".npy"
#    if not ( os.path.exists(path)):
#        print("Generating and saving CBMatrix: " + key )
#        np.save('cb/' + key, cbRecMatrix(key))
#        print("Releasing memory")
#        gc.collect()
#print('done generating matrixes')
#
#finishing initialization dM


#inicialização alternativa
#poe em dM todas as matrizes

#se não se usar o método acima tudo o que se tem de fazer é desomentar isto antes de ligar o servidor
#generateCBMatrix()




#funções auxiliares

def listIDSToJSON(list):
  return jsonify(result = list)

def processFeatures(features, weigths = {}):
    res = []
    defaultWeights = {
            'title'      : 1,
            'actors'     : 0.8,
            'country'    : 0.1,
            'genre'      : 1.1,
            'language'   : 0.5,
            'writer'     : 0.4,
            'plot'       : 0.6,
            'director'   : 0.6, 
            'production' : 0.3
         }    
    for feature,value in features.items():
        if not (weigths[feature]):
            pair = (str(feature),defaultWeights[feature])
        else:
            pair = (str(feature),weigths[feature])
        res.append(pair)
    return res


#inicia as matrizes
#para tal carrega os ficheiros que guardou em cima
def startFeatureMatrixes(features):
    for feature,value in features:
        print("Loading CBMatrix: " + feature)
        dM[feature] = np.load('./cb/' + feature + '.npy') 
        print("Loading Complete")



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




#Devido ao facto que podemos ter muitos dados nas features e pesos, vai ser enviado um JSON
@app.route("/contentBased/<title>/<usage>",  methods=['POST'])
def callCbRecommendations(title,usage):

    req_data = request.get_json()

    unprocessedFeatures = req_data.features
    weigths = req_data.weigths

    features = processFeatures(unprocessedFeatures,weigths)

    #inicia as matrizes
    #só para o caso de se usar o método de ficheiros em FS
    #startFeatureMatrixes(features)

    listRecomended = cbRecFromTitle(str(title),features)
    res = listIDSToJSON(listRecomended)

    dM={}
    return res


@app.route("/collaborativeBased/<int:user>")
def callCfRecommendations(user):
    listRecomended = cfRecommendations(user)
    res = listIDSToJSON(listRecomended)
    return res


@app.route("/hybrid/<int:user>")
def callHibRecomend(user):
  listRecomended = hibRecomend(user)
  res = listIDSToJSON(listRecomended)
  return res





#popular

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
  listRecomended = wsBestRated(str(site))
  res = listIDSToJSON(listRecomended)
  return res
  
