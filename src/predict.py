import pickle
import pandas as pd
from src.feature_engineering import add_features

def predict_sample(input_data):

    # Load model
    with open("models/best_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Convert input to DataFrame
    df = pd.DataFrame([input_data])

    # Apply feature engineering
    df = add_features(df)

    # Predict
    prediction = model.predict(df)

    return prediction[0]
