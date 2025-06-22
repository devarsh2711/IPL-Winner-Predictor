# Step 1: Load and clean the dataset
import pandas as pd

df = pd.read_csv("data/matches.csv")

df = df[df['winner'].notnull()]

teams = [
    'Chennai Super Kings', 'Delhi Capitals', 'Kings XI Punjab',
    'Kolkata Knight Riders', 'Mumbai Indians', 'Rajasthan Royals',
    'Royal Challengers Bangalore', 'Sunrisers Hyderabad'
]

df = df[df['team1'].isin(teams) & df['team2'].isin(teams) & df['winner'].isin(teams)]

df = df[['team1', 'team2', 'city', 'toss_winner', 'toss_decision', 'winner']]


# ✅ Step 2: Encode all categorical columns
from sklearn.preprocessing import LabelEncoder

encoders = {}  # Dictionary to store all encoders

# For each column, fit a LabelEncoder and save it
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le  # Save this encoder for later use in app.py


# Step 3: Train the model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

X = df.drop('winner', axis=1)
y = df['winner']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)


# Step 4: Save model and encoders to disk
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

print("✅ Model and encoders saved successfully!")