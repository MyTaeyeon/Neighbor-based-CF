from flask import Flask, render_template, request
from CF import CF
import pandas as pd

app = Flask(__name__, template_folder='template')

# Load data
columns = ['User_id', 'Movie_id', 'Rating', 'timestamp']
ratings_info = pd.read_csv('./ml-100k/u.data', sep='\t', names=columns, encoding='latin-1')
ratings_matrix = ratings_info.pivot(index='Movie_id', columns='User_id', values='Rating').values

cf = CF(ratings_matrix, 40)
cf.fit()

columns = ['Movie_id', 'Movie_name', 'Year', 'none', 'links', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18']
movies = pd.read_csv("./ml-100k/u.item", sep='|', names=columns, encoding='latin-1')


@app.route('/')
def index():
    recommended_items = cf.recommend(93)
    for idx in range(len(recommended_items)):
        recommended_items[idx]['item'] = movies['Movie_name'][recommended_items[idx]['item']]
    return render_template('index.html', recommended_movies=recommended_items)

if __name__ == '__main__':
    app.run(debug=True)