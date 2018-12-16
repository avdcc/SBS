
#%%
import pandas as pd
import numpy as np
import tkinter 
import matplotlib.pyplot as plt
import re

movies = pd.read_csv('filmes.csv', sep=';', encoding='utf-8')

# ------------------------------------------------------------------------
# Content Based

from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')

# Build a 1-dimensional array with movie titles
titles = movies['title']
indices = pd.Series(movies.index, index=movies['title'])

# Function that get movie recommendations based on the cosine similarity score of movie genres
def recommendationsMatrix(title, feature):

    # Break up the big genre string into a string array and Convert genres to string value
    movies[feature] = movies[feature].str.split(', ')
    movies[feature] = movies[feature].fillna("").astype('str')

    matrix = tfidf_matrixGenre = tf.fit_transform(movies[feature])

    from sklearn.metrics.pairwise import linear_kernel
    cosine_sim = linear_kernel(matrix, matrix)
    return(cosine_sim)

def completeRecommendations(title, features, filters, nRes):

    mx = np.empty([len(titles),len(titles)])
    for f in features:
        mx += recommendationsMatrix(title, f)
        
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
print(
completeRecommendations('Dial M for Murder', ['genre','actors'],['production', 'year'], 100)
)
# "actors";"awards";"country";"director";"genre";"imdb_rating";"imdb_votes";"language";"metascore";"plot";"poster";"production";"ratings";"title";"writer";"year";"dvdYear";"releasedMonth";"releasedYear";"duration"
# "country";"director";"genre";"imdb_rating";"imdb_votes";"language";"metascore";"plot";"poster";"production";"ratings";"title";"writer";"year";"dvdYear";"releasedMonth";"releasedYear";"duration"


# ------------------------------------------------------------------------
# Collaborative Filtering Recommendation

# ratings = pd.read_csv('ratings.csv', sep=';', encoding='utf-8')

# # Fill NaN values in user_id and movie_id column with 0
# ratings['user_id'] = ratings['user_id'].fillna(0)
# ratings['movie_id'] = ratings['movie_id'].fillna(0)

# # Replace NaN values in rating column with average of all values
# ratings['rating'] = ratings['rating'].fillna(ratings['rating'].mean())



# from sklearn.metrics.pairwise import pairwise_distances
# # User Similarity Matrix
# user_correlation = 1 - pairwise_distances(ratings, metric='correlation')
# user_correlation[np.isnan(user_correlation)] = 0
# print(user_correlation[:4, :4])