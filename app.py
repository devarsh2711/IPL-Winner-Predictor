import streamlit as st
import pandas as pd
import pickle

# Load model and encoders
model = pickle.load(open('model.pkl', 'rb'))
encoders = pickle.load(open('encoders.pkl', 'rb'))

# Get options from encoders
teams = sorted([t for t in encoders['team1'].classes_ if isinstance(t, str)])
cities = sorted([c for c in encoders['city'].classes_ if isinstance(c, str)])

# UI
st.title("🏏 IPL Match Winner Predictor")

team1 = st.selectbox("Select Team 1", teams)
team2 = st.selectbox("Select Team 2", teams)
city = st.selectbox("Select Match City", cities)
toss_winner = st.selectbox("Who won the toss?", [team1, team2])
toss_decision = st.selectbox("Toss decision", ['bat', 'field'])

if team1 == team2:
    st.warning("Please select two different teams.")

if st.button("Predict Winner") and team1 != team2:
    # Step 1: Create input DataFrame
    input_df = pd.DataFrame([[team1, team2, city, toss_winner, toss_decision]],
                            columns=['team1', 'team2', 'city', 'toss_winner', 'toss_decision'])

    # Step 2: Encode input using encoders
    for col in input_df.columns:
        input_df[col] = encoders[col].transform(input_df[col])

    # Step 3: Predict probabilities for all teams
    probs = model.predict_proba(input_df)[0]

    # Step 4: Get encoded values for team1 and team2
    team1_encoded = encoders['winner'].transform([team1])[0]
    team2_encoded = encoders['winner'].transform([team2])[0]

    # Step 5: Compare probabilities
    if probs[team1_encoded] > probs[team2_encoded]:
        predicted_team = team1
    else:
        predicted_team = team2

    # Step 6: Show result
    st.success(f"🏆 Predicted Winner: {predicted_team}")