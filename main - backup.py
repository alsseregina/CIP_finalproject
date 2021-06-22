''' 

This program creates for users a personalized movie recommendation based on user's favourite and unfavourite movies.
Personalized recommendation is done through analysis of Netflix open source data based on similarity method. 

'''
import pandas as pd
import numpy as np
from scipy.spatial import distance
import warnings
warnings.filterwarnings("ignore")

def main():
    print('Welcome to Pinewood Movie Reccommender! To get a personalized movie recommendation, we would like to know some of your faviourite and unfavourite ones. Welcome onboard and let''s get started!\n')
    
    #get favourite and unfavourite movies from the user and store them as a list
    fav_movies = get_favourite_movies()
    unfav_movies = get_unfavourite_movies()
    
    print('\nLooking for the perfect movie...')
    
    #uploading open source Netflix data files with movie ratings and meta data to be used for making personalized recommendation
    rating_data = pd.read_csv('data/ratings_small.csv')
    movies_meta = pd.read_csv('data/movies_metadata.csv')
    movies_links = pd.read_csv('data/links.csv')
    
    #merge data sets for further processing:
    movies_links['imdb_id'] = ''
    for i in range(len(movies_links)):
        x = movies_links.iloc[i]
        movies_links['imdb_id'].iloc[i] = 'tt'+(7-len(str(x.imdbId)))*'0'+str(x.imdbId)
        
    movies_links.set_index('imdb_id', inplace = True)
    movies_meta.set_index('imdb_id', inplace = True)
    
    movies_meta = pd.merge(movies_meta, movies_links, right_index = True, left_index = True)
    movies_meta.set_index('movieId', inplace = True)
    
    #Out of meta data we create a separate table that will store the movies ids us index and original movie titles. Movie titles are converted to lower cases to make it easier to match with user input to avoid typos. Note, that user input will be also converted to lower case.
    #This table will be used to (1) get ids of user inputs to include the user in the list and (2) convert the ids of recommendations into rela titles.
    movies_titles = pd.DataFrame(movies_meta['original_title'])
    movies_titles['original_title_lc'] = movies_titles.original_title.str.lower()
    
    # Matrix of ratings (rows = movies, columns = users)
    r = rating_data.pivot(index = 'movieId', columns = 'userId', values = 'rating')
    
    #get movieIds of entered movies by the user and store them in a dictionary where key is movie id and values is the movie
    fav_movies_dict = get_movies_id(fav_movies, movies_titles)
    unfav_movies_dict = get_movies_id(unfav_movies, movies_titles)
    
    #update the matrix of ratings by adding a new user, and then find top 5 similar users based on movies input:
    r, userId = add_new_user(r, fav_movies_dict, unfav_movies_dict)
    similar_users = find_similar_users(r, userId, k = 5)
    
    #make recommendation to the user:
    recommend_movie(r, similar_users, movies_titles)
    
def recommend_movie(r, similar_users, movies_titles):
    if len(similar_users) == 0:
        recommended_movie_id = r.T.mean().dropna().sort_values(ascending = False).index[0]
    else:
        recommended_movie_id = r[similar_users].T.mean().dropna().sort_values(ascending = False).index[0]
    recommended_movie = movies_titles.loc[recommended_movie_id, 'original_title']
    print('\nI recommend '+ recommended_movie)

def get_favourite_movies():
    #get the number of favourite movies from the user, and then the names. The names should be stored as a list that is returned to the main function
    fav_num = -1
    while ((fav_num >= 0) == False):
        try:
            fav_num = int(input('We would like to know some movies that you totally love! How many movies you are ready to mention? If you don''t feel like mentioning any press 0.\n'))
            if fav_num < 0:
                print('Enter a positive number!')
        except ValueError:
            print('Invalid input format. Please enter a number.')
      
    fav_movies = []
    if fav_num > 0:
        for i in range(fav_num):
            msg = 'Enter ' + str(i+1) + ' favourite movie. Make sure that you write the names correctly to helps us make a great recommendation! '
            mov = str(input(msg))
            fav_movies.append(mov)
    print('Found '+str(len(fav_movies))+' of your favorite movies in the database :D')
    return fav_movies

def get_unfavourite_movies():
    #get the number of unfavourite movies from the user, and then the names. The names should be stored as a list that is returned to the main function
    unfav_num = -1
    while ((unfav_num >= 0) == False):
        try:
            unfav_num = int(input('\nWe would like to know some of your unfavourite movies as well. How many movies you are ready to mention? If you don''t feel like mentioning any press 0.\n'))
            if unfav_num < 0:
                print('Enter a positive number!')
        except ValueError:
            print('Invalid input format. Please enter a number.')
            
    unfav_movies = []
    if unfav_num > 0:
        for i in range(unfav_num):
            mov = str(input('Enter ' + str(i+1) + ' unfavourite movie. Make sure that you write the names correctly to helps us make a great recommendation! '))
            unfav_movies.append(mov)
    print('\nFound '+str(len(unfav_movies))+' of your unfavorite movies in my database!')
    return unfav_movies
        
def get_movies_id(movies, movies_titles):
    #Creating a dictinoary to store entered movie id and title together:
    movies_dict = {}
    for movie in movies:
        if movie.lower() in list(movies_titles.original_title_lc):
            movieId = int(movies_titles[movies_titles.original_title_lc == movie.lower()].index[0])
            movies_dict[movie] = movieId
    return movies_dict

def add_new_user(r, fav_movies_dict, unfav_movies_dict):
    #Enter user to matrix with ratings data (r) and set the score "5" for favourite movies and "1" for unfavourite ones
    userId = max(r.columns) + 1
    r[userId] = np.nan
    for movie in fav_movies_dict.keys():
        r.loc[fav_movies_dict[movie], userId] = 5
    for movie in unfav_movies_dict.keys():
        r.loc[unfav_movies_dict[movie], userId] = 1
    return r, userId
    
def find_similar_users(r, userId, k = 5):
    print('Now looking for users with similar taste...')
    #Calculate similarity to discover top 5 similar users to see what kind of movies they like:
    similarities = pd.DataFrame(index = [u for u in r.columns if u != userId], columns = ['distance'])
    for u in r.columns:
        if u != userId:
            if len(r[[userId, u]].dropna()) > 0:
                similarities.loc[u, 'distance'] = distance.euclidean(r[[userId, u]].dropna()[userId], r[[userId, u]].dropna()[u])
    similarities.dropna(inplace = True)
    similarities['similarity'] = 1 - similarities['distance']/similarities['distance'].max()
    
    similar_users = list(similarities.sort_values(by = 'similarity', ascending = False).index[:k])
    return similar_users

main()