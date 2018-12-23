
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

from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')

# Build a 1-dimensional array with movie titles
titles = movies['title']
indices = pd.Series(movies.index, index=movies['title'])
index = pd.Series(movies.index, index=movies['imdb_id'])
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
        # print(mx)
    # print(mx[0])

    sim_scores = list(enumerate(mx[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)  
    sim_scores = list(filter(lambda x: x[0] != idx, sim_scores))

    movie_indices = [i[0] for i in sim_scores]
    return(movie_indices)

def cbRecFromUser(user, features):

    lista = utilizador2Vistos(user)
    mx = np.empty([len(titles),len(titles)])
    for x in range(len(lista)):
        mx[0] += cbRecFromImdb(lista[x], features)

    sim_scores = list(enumerate(mx[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)  
    sim_scores = list(filter(lambda x: x[0] not in lista, sim_scores))

    movie_indices = [i[0] for i in sim_scores]
    return(movie_indices)


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

generateCBMatrix()

print(index2Name(cbRecFromUser(504, [('title',1), ('actors',0.8), ('country',0.1), ('genre',1.1), ('language',0.5), ('writer',0.4),('plot',0.6),('director',0.6), ('production',0.3)]))[0:100]) # 111 514

# print(cbRecFromUser(1,cb))
# a = cbRecFromTitle('Batman: Mystery of the Batwoman', [('title',1), ('actors',0.8), ('country',0.1), ('genre',1.1), ('language',0.5), ('writer',0.4),('plot',0.6),('director',0.6), ('production',0.3)])
# b = cbRecFromTitle('Batman: Mystery of the Batwoman', [('title',1), ('actors',0.8), ('country',0.1), ('genre',1.1), ('language',0.5), ('writer',0.4),('plot',0.6),('director',0.6), ('production',0.3)])
# c = cbRecFromTitle('Batman: Mystery of the Batwoman', [('title',1), ('actors',0.8), ('country',0.1), ('genre',1.1), ('language',0.5), ('writer',0.4),('plot',0.6),('director',0.6), ('production',0.3)])
# d = cbRecFromTitle('Batman: Mystery of the Batwoman', [('title',1), ('actors',0.8), ('country',0.1), ('genre',1.1), ('language',0.5), ('writer',0.4),('plot',0.6),('director',0.6), ('production',0.3)])
# e = cbRecFromTitle('Batman: Mystery of the Batwoman', [('title',1), ('actors',0.8), ('country',0.1), ('genre',1.1), ('language',0.5), ('writer',0.4),('plot',0.6),('director',0.6), ('production',0.3)])

# print((a))

# print((index2Name(a))[0:10])

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

    # o load retorna um tuplo, mas neste caso o pred não tem nada com significado
    pred,svd = dump.load(fileModel)
    vistosLista = utilizador2Vistos(user)

    list = []
    for mov in ratings.imdbId.unique():
        if (mov not in vistosLista):
            list.append((mov, svd.predict(user, mov, 3).est))

    list = sorted(list, key=lambda x: x[1], reverse=True)
    res = [ seq[0] for seq in list ]

    return(res)

# print(cfRecommendations(2))

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
        result = ratings.sort_values(by=[site+ '_rating'], ascending=False)

    return(list(result['imdbId']))

# ------------------------------------------------------------------------
# Hibrido

def hibRecomend(user):
    pass    





# ------------------------------------------------------------------------
# Referencias de Sites?

# http://nbviewer.jupyter.org/github/khanhnamle1994/movielens/blob/master/Deep_Learning_Model.ipynb
# https://www.kaggle.com/rounakbanik/movie-recommender-systems
# https://www.datacamp.com/community/tutorials/recommender-systems-python
# https://www.ethanrosenthal.com/2015/11/02/intro-to-collaborative-filtering/

# ------------------------------------------------------------------------

# "actors";"awards";"country";"director";"genre";"imdbId";"imdb_rating";"imdb_votes";"language";"metascore";"plot";"poster";"production";"ratings";"title";"writer";"year";"dvdYear";"releasedMonth";"releasedYear";"duration"


# lista = ['tt0056172', 'tt0075029', 'tt0118114', 'tt0126029', 'tt0133093', 'tt0373469', 'tt0076759', 'tt0032904', 'tt0109830', 'tt0050083', 'tt0435761', 'tt0044081', 'tt0108052', 'tt0093779', 'tt5027774', 'tt0038650', 'tt0088763', 'tt0048728', 'tt1895587', 'tt0034583', 'tt0118799', 'tt0071360', 'tt0057115', 'tt0770802', 'tt0083833', 'tt4302938', 'tt0423866', 'tt0056095', 'tt0901487', 'tt0469021', 'tt0099851', 'tt1531901', 'tt0042332', 'tt0324080', 'tt8391976', 'tt0094663', 'tt0478049', 'tt0467200', 'tt0188913', 'tt0036154', 'tt0246641', 'tt3903852', 'tt0114287', 'tt0187859', 'tt0056291', 'tt0243017', 'tt0098532', 'tt0377107', 'tt0317740', 'tt0081506', 'tt0405336', 'tt0081070', 'tt0092494', 'tt0057197', 'tt0067741', 'tt0780571', 'tt0107808', 'tt0077292', 'tt3201722', 'tt0097889', 'tt0109348', 'tt0066817', 'tt0243609', 'tt0491152', 'tt0181627', 'tt0092263', 'tt0450405', 'tt0119008', 'tt0119887', 'tt0095488', 'tt0110889', 'tt0058083', 'tt0114787', 'tt0805570', 'tt0098663', 'tt0120148', 'tt0099699', 'tt0468445', 'tt0304262', 'tt0093200', 'tt0395125', 'tt0117509', 'tt0035753', 'tt2166616', 'tt0089869', 'tt0110917', 'tt0310775', 'tt0141369', 'tt0120185', 'tt0108255', 'tt0211443', 'tt0120179', 'tt0118615', 'tt0120685', 'tt0118688', 'tt0185183']

# filtros = {'gen':'Romance'}

# ------------------------------------------------------------------------
# generatezCBMatrix()