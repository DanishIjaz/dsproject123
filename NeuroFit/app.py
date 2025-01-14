import streamlit as st
import tensorflow as tf
from utils.preprocess import preprocess_user_data
import pickle

# Load models
diet_model = tf.keras.models.load_model('models/diet_recommender.h5')
workout_model = tf.keras.models.load_model('models/workout_recommender.h5')

# Load label encoders
with open('models/diet_label_encoder.pkl', 'rb') as file:
    diet_label_encoder = pickle.load(file)

with open('models/workout_label_encoder.pkl', 'rb') as file:
    workout_label_encoder = pickle.load(file)

# Prediction Functions
def recommend_diet(age, height, weight, goal):
    user_data = preprocess_user_data(age, height, weight, goal)
    prediction = diet_model.predict([user_data])
    return diet_label_encoder.inverse_transform([prediction.argmax()])[0]

def recommend_workout(age, height, weight, goal):
    user_data = preprocess_user_data(age, height, weight, goal)
    prediction = workout_model.predict([user_data])
    return workout_label_encoder.inverse_transform([prediction.argmax()])[0]

# Streamlit UI
st.title("NeuroFit - Personalized Fitness Recommendations")

# Collect user inputs
age = st.number_input("Age", min_value=10, max_value=100, value=25)
height = st.number_input("Height (in cm)", min_value=100, max_value=250, value=170)
weight = st.number_input("Weight (in kg)", min_value=30, max_value=200, value=70)
goal = st.selectbox("Goal", ["Lose Weight", "Maintain Weight", "Gain Muscle", "Increase Strength"])

# Recommend Diet
if st.button("Recommend Diet"):
    recommended_diet = recommend_diet(age, height, weight, goal)
    st.write(f"Recommended Diet: {recommended_diet}")

# Recommend Workout
if st.button("Recommend Workout"):
    recommended_workout = recommend_workout(age, height, weight, goal)
    st.write(f"Recommended Workout: {recommended_workout}")
