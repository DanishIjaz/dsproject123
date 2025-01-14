import tensorflow as tf
from utils.preprocess import preprocess_user_data

# Load models
diet_model = tf.keras.models.load_model("models/diet_recommender.h5")
workout_model = tf.keras.models.load_model("models/workout_recommender.h5")

def recommend_diet(age, height, weight, goal):
    # Preprocess user data for diet recommendation
    user_data = preprocess_user_data(age, height, weight, goal)
    
    # Predict the diet recommendation (you may want to add batch dimension for prediction)
    prediction = diet_model.predict([user_data])
    
    # Decode the prediction into a human-readable result
    return decode_diet_prediction(prediction)

def recommend_workout(age, height, weight, goal):
    # Preprocess user data for workout recommendation
    user_data = preprocess_user_data(age, height, weight, goal)
    
    # Predict the workout recommendation (same batch dimension issue)
    prediction = workout_model.predict([user_data])
    
    # Decode the prediction into a human-readable result
    return decode_workout_prediction(prediction)

def decode_diet_prediction(prediction):
    # Assuming the model returns probabilities for each class (diet type)
    diets = ["Keto", "Vegan", "Balanced", "High Protein"]
    
    # If the prediction is a list of probabilities, take the one with the highest probability
    predicted_class_index = prediction.argmax(axis=-1)  # Get the index with the highest probability
    return diets[predicted_class_index[0]]  # Return the corresponding diet

def decode_workout_prediction(prediction):
    # Assuming the model returns probabilities for each class (workout type)
    workouts = ["Cardio", "Strength Training", "Yoga", "HIIT"]
    
    # If the prediction is a list of probabilities, take the one with the highest probability
    predicted_class_index = prediction.argmax(axis=-1)  # Get the index with the highest probability
    return workouts[predicted_class_index[0]]  # Return the corresponding workout
