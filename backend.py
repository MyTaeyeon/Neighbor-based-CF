from flask import Flask, render_template, request
from CF import CF
import pandas as pd

app = Flask(__name__, template_folder='templates')

# Load data
columns = ['User_id', 'Movie_id', 'Rating', 'timestamp']
test_data = pd.read_csv('./ml-100k/u.data', sep='\t', names=columns, encoding='latin-1')
testing_data = test_data.pivot(index='Movie_id', columns='User_id', values='Rating').values

cf = CF(testing_data)
cf.fit()

movie_info = {}
with open("ml-100k/u.item", encoding="latin-1") as f:
    for line in f:
        data = line.split("|")
        movie_info[int(data[0])] = {
            "title": data[1],
            "rating": 0  # Initialize rating to 0
        }

# Load ratings data (if available) and update movie_info
with open("ml-100k/u.data") as f:
    for line in f:
        user_id, movie_id, rating, _ = map(int, line.split("\t"))
        movie_info[movie_id]["rating"] += rating

# Calculate average rating for each movie
for movie_id, info in movie_info.items():
    if info["rating"] > 0:
        info["rating"] /= 100  # Assuming each movie has 100 ratings on average

@app.route('/')
def index():
    recommended_items = cf.recommend(100)
    recommended_movies = []
    for movie_id in recommended_items:
        recommended_movies.append(movie_info[movie_id])
    show_recommended_movies = True
    return render_template('index.html', recommended_movies=recommended_movies, show_recommended_movies=show_recommended_movies)

@app.route('/search', methods=['GET', 'POST'])
def search():
    query = request.form['query']
    results = []
    for movie_id, info in movie_info.items():
        if query.lower() in info['title'].lower():
            results.append(info)
    return render_template('index.html', query=query, results=results)

if __name__ == '__main__':
    app.run(debug=True)