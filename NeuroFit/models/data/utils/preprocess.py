def preprocess_user_data(age, height, weight, goal):
    # Normalize user data for neural network input
    return [age / 100, height / 250, weight / 200, goal_mapping(goal)]

def goal_mapping(goal):
    return {"Lose Weight": 0, "Gain Muscle": 1, "Stay Fit": 2}.get(goal, -1)
