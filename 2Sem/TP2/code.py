
import datetime
import pandas as pd
import numpy as np
import tkinter 
import matplotlib.pyplot as plt
import re

import pickle
import pprint
import sys


import matplotlib.pyplot as plt
from scipy import stats

# ------------------------------------------------------------------------
# import seaborn as sns

# from ast import literal_eval
# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
# from nltk.stem.snowball import SnowballStemmer
# from nltk.stem.wordnet import WordNetLemmatizer
# from nltk.corpus import wordnet
# from surprise import Reader, Dataset, SVD, evaluate, dump

# import warnings; warnings.simplefilter('ignore')

city = "Guimaraes"

# ------------------------------------------------------------------------
# FLOW

def dateTimeFixer(par):

  par = par.split("___")
  # flag = False
  par[0], flag = timeFixer(par[0])

  if flag:
    par[1] = acresDate(par[1])
  # print(par)
  return (str(par[0]) + "___" + str(par[1]))

def acresDate (date):

  datetime_object = datetime.datetime.strptime(date, '%Y-%m-%d')
  datetime_object += datetime.timedelta(days=1)

  return(str(datetime_object).split(" ")[0])

def timeFixer(time):

  lista = time.split(':')

  lista = list(map(int, lista))
  flagDia = False

  # Fix Segundos (0/60)
  if (lista[2] > 30):
    lista[1] += 1
  lista[2] = "00"

  # Fix Minutos (0/15/30/45)
  if (lista[1] < 7):
    lista[1] = "00"
  elif (lista[1] < 23):
    lista[1] = "15"
  elif (lista[1] < 37):
    lista[1] = "30"
  elif (lista[1] < 53):
    lista[1] = "45"
  else:
    lista[1] = "00"
    lista[0] += 1

  # Fix Horas 
  if (lista[0] == 0):
    lista[0] = "00"
  elif(lista[0] == 24):
    flagDia = True

  return (str(lista[0]) + ":" + lista[1]), flagDia

def stackSplit(time, sep, rev=False):
  if rev:
    return (time.rsplit(sep,1)[0])
  return (time.split(sep,1)[0])

def cleanRepeated(inputFile, outputFile):
  lines_seen = set() 
  outfile = open(inputFile, "w")

  for line in open(outputFile, "r"):
    if line not in lines_seen:
      outfile.write(line)
      lines_seen.add(line)
  outfile.close()

def tFlow():

  data = pd.read_csv(city + '/tf.csv', sep=',', encoding='utf-8')

  # ----------
  # Verificar se a informacao sobre as estradas faz senso
  # streatNumbers = data.road_num.unique()
  # for i in streatNumbers:
  #   print(data.query('road_num==%i' % i)['road_name'].unique())
  #   print(data.query('road_num==%i' % i)['functional_road_class_desc'].unique())

  # ----------
  # Split de YYYY-MM-DD HH:MM:SS.MMMMMM para colunas separadas
  data['dateComplete'] = data['creation_date'].apply(stackSplit, args=('.',))
  data['creation_date'], data['creation_time'] = data['creation_date'].str.split(' ', 1).str
  data['creation_time'] = data['creation_time'].apply(stackSplit, args=('.',))

  data['creation'] = data['creation_time'] + "___" + data['creation_date']
  data['creation'] = data['creation'].apply(dateTimeFixer)

  data['creation_date'], data['creation_time'] = data['creation'].str.split('___', 1).str


  # ----------
  # Eliminar Inutil
  del data['city_name'] # Inutil
  del data['road_num'] # Inutil
  del data['creation'] # Inutil

  # ----------
  # Save2File
  data.to_csv(city + "/modtf.csv", sep=',', encoding='utf-8', index=False)

# ------------------------------------------------------------------------
# WEATHER

def func(descriptions):

  # ['céu pouco nublado','nuvens dispersas','nuvens quebradas', 'algumas nuvens','céu claro','chuva fraca','tempo nublado','chuva moderada','nuvens quebrados','neblina','névoa','chuva leve','garoa fraca','chuva','trovoada','trovoada com chuva leve','trovoada com chuva','chuva de intensidade pesado', 'chuva de intensidade pesada'] 
  lista = []

  for desc in descriptions:
    if desc in ['céu pouco nublado','nuvens dispersas','nuvens quebradas', 'algumas nuvens','céu claro','tempo nublado','nuvens quebrados']:
      lista.append(0)
    elif desc in ['chuva fraca','tempo nublado','chuva moderada','neblina','névoa','chuva leve','garoa fraca','chuva']:
      lista.append(1)
    elif desc in ['trovoada','trovoada com chuva leve','trovoada com chuva','chuva de intensidade pesado', 'chuva de intensidade pesada']:
      lista.append(2)
    else:
      lista.append(-100)
  return(lista)

def tWeather():

  data = pd.read_csv(city + '/w.csv', sep=',', encoding='utf-8')

  data['dateComplete'] = data['creation_date'].apply(stackSplit, args=('.',))

  # ----------
  # Split Data
  data['sunrise_date'], data['sunrise_time'] = data['sunrise'].str.split(' ', 1).str
  data['sunrise_time'] = data['sunrise_time'].apply(stackSplit, args=('.',))
  # data['sunrise_time'] = data['sunrise_time'].apply(stackSplit, args=(':',True))

  data['sunset_date'], data['sunset_time'] = data['sunset'].str.split(' ', 1).str
  data['sunset_time'] = data['sunset_time'].apply(stackSplit, args=('.',))
  # data['sunset_time'] = data['sunset_time'].apply(stackSplit, args=(':',True))

  data['creation_date'], data['creation_time'] = data['creation_date'].str.split(' ', 1).str
  data['creation_time'] = data['creation_time'].apply(stackSplit, args=('.',))
  # data['creation_time'] = data['creation_time'].apply(stackSplit, args=(':',True))

  
  
  
  data['creation'] = data['creation_time'] + "___" + data['creation_date']
  data['creation'] = data['creation'].apply(dateTimeFixer)

  data['creation_date'], data['creation_time'] = data['creation'].str.split('___', 1).str


  del data['sunrise'] # Inutil
  del data['sunset'] # Inutil
 

  # ----------
  # State
  data['state'] = func(data['weather_description']) 
  del data['rain'] # Inutil

  # ----------
  # Eliminar Inutil
  del data['city_name'] # Inutil
  # del data['weather_description']
  del data['creation']


  del data['sunset_date'] # Inutil
  del data['sunset_time'] # Inutil
  del data['sunrise_date'] # Inutil
  del data['sunrise_time'] # Inutil

  # ----------
  # Save2File
  data.to_csv(city + "/modw.csv", sep=',', encoding='utf-8', index=False)

# ------------------------------------------------------------------------
# INCIDENTS

def tIncidents():
  data = pd.read_csv(city + '/ti.csv', sep=',', encoding='utf-8')

  data['dateComplete'] = data['incident_date']
  # ----------
  # Split Data
  data['incident_date'], data['incident_time'] = data['incident_date'].str.split(' ', 1).str
  data['incident_time'] = data['incident_time'].str.split('.', 1)[0][0]
  # data['incident_time'] = data['incident_time'].str.rsplit(':', 1)[0][0]


  data['creation'] = data['incident_time'] + "___" + data['incident_date']
  data['creation'] = data['creation'].apply(dateTimeFixer)

  data['incident_date'], data['incident_time'] = data['creation'].str.split('___', 1).str

  # ----------
  # Eliminar Inutil
  del data['city_name'] # Inutil

  del data['creation'] # Inutil
  del data['incident_time'] # Inutil
  del data['incident_date'] # Inutil


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

def checkInvalid(dSet='/tf.csv'):

  df = pd.read_csv(city + '/w.csv', sep=',', encoding='utf-8')
  # data = open(city + dSet)
  # linhas = data.read()

  # print(linhas)

  null_data = df[df.isnull().any(axis=1)]

  print(null_data)
  # aux = list(data)

# ------------------------------------------------------------------------
 
def joiner():

  datatf = pd.read_csv(city + '/modtf.csv', sep=',', encoding='utf-8')
  dataw = pd.read_csv(city + '/modw.csv', sep=',', encoding='utf-8')


  datatf['dateComplete'] = pd.to_datetime(datatf['dateComplete'])
  dataw['dateComplete'] = pd.to_datetime(dataw['dateComplete'])

  datatf.sort_values(by=['dateComplete'], inplace=True)
  dataw.sort_values(by=['dateComplete'], inplace=True)

  merged_df = pd.merge_asof( datatf, dataw, left_on=['dateComplete'], right_on=['dateComplete'], direction='nearest')

  merged_df.drop(columns=["creation_date_y","creation_time_y"])
  merged_df.rename(columns={'creation_date_x':'creation_date', 'creation_time_x':'creation_time'}, inplace=True)


  # merged_df = pd.merge_asof(datatf,dataw,right_index=True,left_index=True,direction='nearest',tolerance=tol)
  merged_df.to_csv(city + "/tfw.csv", sep=',', encoding='utf-8', index=False)


# Transformacoes Necessarias

# cleanRepeated("./Guimaraes/w.csv")

# print(acresDate("2019-02-28"))

# tFlow()
# tWeather()
# tIncidents()
# tDatas()
# checkInvalid()
joiner()








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

