from flask import Flask,jsonify,request

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


regex = re.compile('[^a-zA-Z]')
stemmer = SnowballStemmer('english')

from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')


movies = pd.read_csv('filmes.csv', sep='§', encoding='utf-8')
ratings = pd.read_csv('movielens.csv', sep=';', encoding='utf-8')

titles = movies['imdb_id']
indices = pd.Series(movies.index, index=movies['title'])
index = pd.Series(movies.index, index=movies['imdb_id'])

cb = ['title', 'actors', 'country', 'genre', 'language', 'writer', 'plot', 'director', 'production']
dM = np.empty([len(titles),len(titles)])

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


def cbRecMatrix(feature):
    
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






def cbRecFromTitle(title):
    idx = indices[title]
    return (cbRecFromId(idx))


def cbRecFromId(idx):

    mx = np.empty([len(titles),len(titles)])
    mx[0] = (dM[idx])

    sim_scores = list(enumerate(mx[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)  
    sim_scores = list(filter(lambda x: x[0] != idx, sim_scores))

    movie_indices = [i[0] for i in sim_scores]
    return(index2imdbId(movie_indices))


def cbRecFromImdb(title):
    idx = index[title]
    mx = np.empty([len(titles),len(titles)])
    mx[0] += (dM[idx])

    return(mx[0])


def cbRecFromUser(user):

    lista = utilizador2Vistos(user)
    mx = np.empty([len(titles),len(titles)])
    for x in range(len(lista)):
        mx[0] += cbRecFromImdb(lista[x])

    sim_scores = list(enumerate(mx[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)  
    sim_scores = list(filter(lambda x: x[0] not in lista, sim_scores))

    movie_indices = [i[0] for i in sim_scores]
    return(index2imdbId(movie_indices))


# ------------------------------------------

import gc

def generateCBMatrix(dM,att):
    for key in att:
        gc.collect()
        print("Generating CBMatrix: " + key[0])
        dM += cbRecMatrix(key[0]) * key[1] 
    print("Geration Complete")



# ------------------------------------------------------------------------
# Collaborative Filtering Recommendation

fileModel = 'testeOutModel'

def startPredModel(ratings,fileOutput):

    reader = Reader ()

    data = Dataset.load_from_df(ratings[['userId', 'imdbId', 'rating']], reader)
    data.split(n_folds=5) # 5

    svd = SVD()
    evaluate(svd, data, measures=['RMSE', 'MAE'])

    trainset = data.build_full_trainset()
    svd.train(trainset)

    dump.dump(fileOutput,None,svd,1)

def  loadFileModel(file_name, encoding='ASCII'):

    dump_obj = pickle.load(open(file_name, 'rb'), encoding=encoding)

    return dump_obj['predictions'], dump_obj['algo']


def cfRecommendations(user):

    pred,svd = loadFileModel(str(fileModel),encoding='latin1')
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

    ratings = pd.read_csv('filmes.csv', sep='§', encoding='utf-8')
    if (site == 'meta'):
        result = ratings.sort_values(by=['Metacritic'], ascending=False)
    elif(site=='imdb'):    
        result = ratings.sort_values(by=['imdb_rating'], ascending=False)
    elif(site=='IMD'):    
        result = ratings.sort_values(by=['Internet Movie Database'], ascending=False)
    elif(site=='rotten'):    
        result = ratings.sort_values(by=['Rotten tomatoes'], ascending=False)
    elif(site=='metascore'):
        result = ratings.sort_values(by=['metascore'], ascending=False)

    return(list(result['imdb_id']))

# ------------------------------------------------------------------------
# Hibrido

def hibRecomend(user):
    l = []
    
    l.append(cbRecFromUser(user))
    l.append(cfRecommendations(user))
    l.append(many2One(
        [
            userBestRated(), 
            userMostPopular(),
            wsBestRated('meta'),
            wsBestRated('imdb'),
            wsBestRated('IMD'),
            wsBestRated('rotten'),
            wsBestRated('metascore')
        ]
    ))
    

    return (many2One(l)) 






#end of code from teste.py

#inicialização da matriz

generateCBMatrix(dM,
                [('title',1), ('actors',0.8), ('country',0.1), 
                ('genre',1.1), ('language',0.5), ('writer',0.4),('plot',0.6),
                ('director',0.6), ('production',0.3)]
                )


startPredModel(ratings,fileModel)


print("  * JS web frontend running on http://127.0.0.1:3000/")


#funções auxiliares

def listIDSToJSON(list):
  return jsonify(result = list)





#caminhos do servidor

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




@app.route("/contentBased/<titleOrId>/<usage>")
def callCbRecommendations(titleOrId,usage):

    if(usage=='title'):
        listRecomended = cbRecFromTitle(str(titleOrId))
    else:
        listRecomended = cbRecFromUser(int(titleOrId))
    
    res = listIDSToJSON(listRecomended)

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
  
