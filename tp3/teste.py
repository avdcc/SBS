
#%%
import pandas as pd
import numpy as np
import tkinter 
import matplotlib.pyplot as plt
import re

movies = pd.read_csv('filmes.csv', sep=';', encoding='utf-8')

# ------------------------------------------------------------------------
# Gera sugestoes com base em features de um filme

from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')

# Build a 1-dimensional array with movie titles
titles = movies['title']
indices = pd.Series(movies.index, index=movies['title'])

# Function that get movie recommendations based on the cosine similarity score of movie genres
def recommendations(title,feature):

    # Break up the big genre string into a string array and Convert genres to string value
    movies[feature] = movies[feature].str.split(', ')
    movies[feature] = movies[feature].fillna("").astype('str')

    matrix = tfidf_matrixGenre = tf.fit_transform(movies[feature])

    from sklearn.metrics.pairwise import linear_kernel
    cosine_sim = linear_kernel(matrix, matrix)

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    movie_indices = [i[0] for i in sim_scores]
    print (type(titles.iloc[movie_indices]))
    return titles.iloc[movie_indices]

# - Print Resultados

print(recommendations('Toy Story 2', 'genre').head(20))
print(recommendations('Toy Story 2', 'actors').head(20))

# ------------------------------------------------------------------------
