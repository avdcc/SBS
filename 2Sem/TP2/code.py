
import datetime

import pandas as pd
import numpy as np
import tkinter 
import matplotlib.pyplot as plt
import re

import matplotlib.pyplot as plt
from scipy import stats
# import seaborn as sns

# from ast import literal_eval
# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
# from nltk.stem.snowball import SnowballStemmer
# from nltk.stem.wordnet import WordNetLemmatizer
# from nltk.corpus import wordnet
# from surprise import Reader, Dataset, SVD, evaluate, dump

import pickle

import pprint
# import warnings; warnings.simplefilter('ignore')

# ------------------------------------------------------------------------
# FLOW

def tFlow():

  data = pd.read_csv(city + '/tf.csv', sep=',', encoding='utf-8')

  # ----------
  # Verificar se a informacao sobre as estradas faz senso
  streatNumbers = data.road_num.unique()
  for i in streatNumbers:
    print(data.query('road_num==%i' % i)['road_name'].unique())
    print(data.query('road_num==%i' % i)['functional_road_class_desc'].unique())

  # ----------
  # Split de YYYY-MM-DD HH:MM:SS.MMMMMM para colunas separadas
  data['creation_date'], data['creation_time'] = data['creation_date'].str.split(' ', 1).str
  data['creation_time'] = data['creation_time'].str.split('.', 1)[0][0]

  # ----------
  # Eliminar Inutil
  del data['city_name'] # Inutil
  del data['road_name'] # Inutil

  # ----------
  # Save2File
  data.to_csv(city + "/modtf.csv", sep=',', encoding='utf-8', index=False)

# ------------------------------------------------------------------------
# WEATHER

def tWeather():

  data = pd.read_csv(city + '/w.csv', sep=',', encoding='utf-8')

  # ----------
  # Split Data
  data['sunrise_date'], data['sunrise_time'] = data['sunrise'].str.split(' ', 1).str
  data['sunrise_time'] = data['sunrise_time'].str.split('.', 1)[0][0]

  data['sunset_date'], data['sunset_time'] = data['sunset'].str.split(' ', 1).str
  data['sunset_time'] = data['sunset_time'].str.split('.', 1)[0][0]

  data['creation_date'], data['creation_time'] = data['creation_date'].str.split(' ', 1).str
  data['creation_time'] = data['creation_time'].str.split('.', 1)[0][0]
  
  # ----------
  # Eliminar Inutil
  del data['city_name'] # Inutil

  # ----------
  # Save2File
  data.to_csv(city + "/modw.csv", sep=',', encoding='utf-8', index=False)

# ------------------------------------------------------------------------
# INCIDENTS

def tIncidents():

  data = pd.read_csv(city + '/ti.csv', sep=',', encoding='utf-8')

  # ----------
  # Split Data
  data['incident_date'], data['incident_time'] = data['incident_date'].str.split(' ', 1).str
  data['incident_time'] = data['incident_time'].str.split('.', 1)[0][0]
  
  # ----------
  # Eliminar Inutil
  del data['city_name'] # Inutil

  # ----------
  # Save2File
  data.to_csv(city + "/modti.csv", sep=',', encoding='utf-8', index=False)

# ------------------------------------------------------------------------
# DATAS

def tDatas():

  data = pd.read_csv(city + '/modtf.csv', sep=',', encoding='utf-8')
  uniq = data.creation_date.unique()

  dias = {'data': uniq}
  df = pd.DataFrame(data=dias)

  # df['sem'] = datetime.datetime.strptime(df['data'], '%Y-%m-%d').weekday()
  df['sem'] = df.apply(lambda row: datetime.datetime.strptime(row['data'], '%Y-%m-%d').weekday(), axis=1)
  
  df.to_csv(city + "/datas.csv", sep=',', encoding='utf-8', index=False)

  # FALTA ADICIONAR OS FERIADOS


# ------------------------------------------------------------------------

city = "Guimaraes"

# Transformacoes Necessarias

# tFlow()
# tWeather()
# tIncidents()

tDatas()



# ------------------------------------------------------------------------


























# ------------------------------------------------------------------------



# regex = re.compile('[^a-zA-Z]')
# stemmer = SnowballStemmer('english')

# from sklearn.feature_extraction.text import TfidfVectorizer
# tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')

# # Build a 1-dimensional array with movie titles
# titles = movies['imdb_id']
# indices = pd.Series(movies.index, index=movies['title'])
# index = pd.Series(movies.index, index=movies['imdb_id'])
# # print(index)

# cb = ['title', 'actors', 'country', 'genre', 'language', 'writer', 'plot', 'director', 'production']
# dM = np.empty([len(titles),len(titles)])

# # Fill NaN values in user_id and movie_id column with 0
# ratings['userId'] = ratings['userId'].fillna(0)
# ratings['imdbId'] = ratings['imdbId'].fillna(0)

# # Replace NaN values in rating column with average of all values
# ratings['rating'] = ratings['rating'].fillna(ratings['rating'].mean())

# ------------------------------------------------------------------------

