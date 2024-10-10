
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

app = Flask(__name__,template_folder='templates')

# Sample data (assuming you have a CSV with relevant data)
data = pd.read_csv('data.csv')
irrelevant = ['mid', 'date', 'venue', 'batsman', 'bowler', 'striker', 'non-striker']
data = data.drop(irrelevant, axis=1)

const_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
               'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
               'Delhi Daredevils', 'Sunrisers Hyderabad']
data = data[(data['batting_team'].isin(const_teams)) & (data['bowling_team'].isin(const_teams))]

le = LabelEncoder()
for col in ['batting_team', 'bowling_team']:
    data[col] = le.fit_transform(data[col])

columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0, 1])], remainder='passthrough')
data = np.array(columnTransformer.fit_transform(data))

cols = ['batting_team_Chennai Super Kings', 'batting_team_Delhi Daredevils', 'batting_team_Kings XI Punjab',
        'batting_team_Kolkata Knight Riders', 'batting_team_Mumbai Indians', 'batting_team_Rajasthan Royals',
        'batting_team_Royal Challengers Bangalore', 'batting_team_Sunrisers Hyderabad', 
        'bowling_team_Chennai Super Kings', 'bowling_team_Delhi Daredevils', 
        'bowling_team_Kings XI Punjab', 'bowling_team_Kolkata Knight Riders',
        'bowling_team_Mumbai Indians', 'bowling_team_Rajasthan Royals',
        'bowling_team_Royal Challengers Bangalore', 'bowling_team_Sunrisers Hyderabad', 
        'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5', 'total']

df = pd.DataFrame(data, columns=cols)
features = df.drop(['total'], axis=1)
labels = df['total']

# Split data
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.20, shuffle=True)

# Train model
forest = RandomForestRegressor()
forest.fit(train_features, train_labels)

# Function for predicting the score
def predict_score(batting_team, bowling_team, runs, wickets, overs, runs_last_5, wickets_last_5, model=forest):
    prediction_array = []
    
    # One-hot encode batting team
    teams = ['Chennai Super Kings', 'Delhi Daredevils', 'Kings XI Punjab', 'Kolkata Knight Riders',
             'Mumbai Indians', 'Rajasthan Royals', 'Royal Challengers Bangalore', 'Sunrisers Hyderabad']
    
    for team in teams:
        if team == batting_team:
            prediction_array.append(1)
        else:
            prediction_array.append(0)

    # Bowling team encoding
    for team in teams:
        if team == bowling_team:
            prediction_array.append(1)
        else:
            prediction_array.append(0)

    # Add numeric features
    prediction_array += [runs, wickets, overs, runs_last_5, wickets_last_5]

    # Convert list to numpy array and reshape
    prediction_array = np.array([prediction_array])

    # Predict the score using the trained model
    pred = model.predict(prediction_array)
    return int(round(pred[0]))

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        # Get form data
        batting_team = request.form['batting_team']
        bowling_team = request.form['bowling_team']
        runs = int(request.form['runs'])
        wickets = int(request.form['wickets'])
        overs = float(request.form['overs'])
        runs_last_5 = int(request.form['runs_last_5'])
        wickets_last_5 = int(request.form['wickets_last_5'])

        # Get prediction
        prediction = predict_score(batting_team, bowling_team, runs, wickets, overs, runs_last_5, wickets_last_5)

    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
