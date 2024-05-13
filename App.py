from flask import Flask, render_template, request
from CF import CF
import pandas as pd
import numpy as np

app = Flask(__name__, template_folder='template')

# Load data
columns = ['User_id', 'Movie_id', 'Rating', 'timestamp']
ratings_info = pd.read_csv('./ml-100k/u.data', sep='\t', names=columns, encoding='latin-1')
ratings_matrix = ratings_info.pivot(index='Movie_id', columns='User_id', values='Rating').values

columns = ['Movie_id', 'Movie_name', 'Year', 'none', 'links', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18']
movies = pd.read_csv("./ml-100k/u.item", sep='|', names=columns, encoding='latin-1')

last_movie = 0
user_id = 15
genre = ("Unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western")

@app.route('/detail/<string:movie_index>')
def movie_detail(movie_index):
    global last_movie, movies, genre, ratings_matrix
    movie_index = int(movie_index)
    last_movie = movie_index
    movie_name = movies.iloc[movie_index]['Movie_name']
    rate = 0 if np.isnan(ratings_matrix[last_movie][user_id]) else ratings_matrix[last_movie][user_id]
    tags = [genre[i] for i in range(19) if movies[str(i)][movie_index] == 1]

    cnt = 0
    for x in ratings_matrix[last_movie]:
        if not np.isnan(x): cnt+=1

    return render_template('movie_detail.html', movie_name=movie_name, tags=tags, rate=rate, numbers=cnt)

@app.route('/rate', methods=['POST'])
def rate_movie():
    global ratings_matrix, last_movie, user_id
    if request.method == 'POST':
        rate_score = request.form['rate_score']
        ratings_matrix[last_movie][user_id] = int(rate_score)
        return "Đã nhận được đánh giá: " + rate_score
    
@app.route('/search', methods=['POST'])
def search_movie():
    global movies
    if request.method == 'POST':
        search_tag = request.form['search']
        for idx, i in enumerate(genre):
            if search_tag[1:].lower() == i.lower():
                search_tag = str(idx)
                break
        movies_name = []
        for idx, i in enumerate(movies[search_tag]): 
            if i == 1:
                g = dict()
                g['id'] = idx
                g['item'] = movies.loc[idx, 'Movie_name']
                g['tags'] = [genre[j] for j in range(19) if movies[str(j)][idx] == 1]
                movies_name.append(g)

        return render_template('search.html', movies_name = movies_name)
    
@app.route('/history')
def history():
    global user_id, movies, ratings_matrix
    items = []
    cnt = 1
    for idx, x in enumerate(ratings_matrix[:, user_id]):
        if not np.isnan(x):
            g = dict()
            g['index'] = cnt
            cnt += 1
            g['item'] = movies.loc[idx, 'Movie_name']
            g['tags'] = [genre[i] for i in range(19) if movies[str(i)][idx] == 1]
            g['rating'] = x
            items.append(g)
    return render_template('history.html', rated_movies=items)

@app.route('/')
def index():
    global user_id
    cf = CF(ratings_matrix, 40)
    cf.fit()
    recommended_items = cf.recommend(user_id)
    for idx in range(len(recommended_items)):
        recommended_items[idx]['id'] = recommended_items[idx]['index']
        recommended_items[idx]['index'] = idx+1
        recommended_items[idx]['item'] = movies['Movie_name'][recommended_items[idx]['id']]
        recommended_items[idx]['tags'] = [genre[i] for i in range(19) if movies[str(i)][recommended_items[idx]['id']] == 1]
    return render_template('index.html', recommended_movies=recommended_items)

if __name__ == '__main__':
    app.run(debug=True)