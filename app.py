from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

app = Flask(__name__,template_folder='templates')

# Load and preprocess data
data = pd.read_csv('data.csv')
irrelevant = ['mid', 'date', 'venue', 'batsman', 'bowler', 'striker', 'non-striker']
data = data.drop(irrelevant, axis=1)
const_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals', 'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
                'Delhi Daredevils', 'Sunrisers Hyderabad']
data = data[(data['batting_team'].isin(const_teams)) & (data['bowling_team'].isin(const_teams))]
le = LabelEncoder()
for col in ['batting_team', 'bowling_team']:
    data[col] = le.fit_transform(data[col])
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(),  [0, 1])],  remainder='passthrough')
data = np.array(columnTransformer.fit_transform(data))
cols = ['batting_team_Chennai Super Kings', 'batting_team_Delhi Daredevils', 'batting_team_Kings XI Punjab',
        'batting_team_Kolkata Knight Riders', 'batting_team_Mumbai Indians', 'batting_team_Rajasthan Royals',
        'batting_team_Royal Challengers Bangalore', 'batting_team_Sunrisers Hyderabad',
        'bowling_team_Chennai Super Kings', 'bowling_team_Delhi Daredevils', 'bowling_team_Kings XI Punjab',
        'bowling_team_Kolkata Knight Riders', 'bowling_team_Mumbai Indians', 'bowling_team_Rajasthan Royals',
        'bowling_team_Royal Challengers Bangalore', 'bowling_team_Sunrisers Hyderabad', 'runs', 'wickets', 'overs',
        'runs_last_5', 'wickets_last_5', 'total']
df = pd.DataFrame(data, columns=cols)
features = df.drop(['total'], axis=1)
labels = df['total']
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.20, shuffle=True)
forest = RandomForestRegressor()
forest.fit(train_features, train_labels)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        batting_team = request.form['batting_team']
        bowling_team = request.form['bowling_team']
        overs = float(request.form['overs'])
        runs = float(request.form['runs'])
        wickets = int(request.form['wickets'])
        runs_last_5 = float(request.form['runs_last_5'])
        wickets_last_5 = int(request.form['wickets_last_5'])

        # Preprocess input data
        batting_team_encoded = le.transform([batting_team])[0]
        bowling_team_encoded = le.transform([bowling_team])[0]
        input_data = np.array([batting_team_encoded, bowling_team_encoded, runs, wickets, overs, runs_last_5, wickets_last_5])
        input_data = columnTransformer.transform([input_data])
        predicted_score = forest.predict(input_data)[0]

        return render_template('index.html', predicted_score=predicted_score, 
                               batting_team=batting_team, bowling_team=bowling_team, 
                               overs=overs, runs=runs, wickets=wickets, 
                               runs_last_5=runs_last_5, wickets_last_5=wickets_last_5)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
