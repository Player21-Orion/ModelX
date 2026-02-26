import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from src.feature_engineering import add_features

def train_models(df):

    df = add_features(df)

    X = df.drop("kpt", axis=1)
    y = df["kpt"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(),
        "XGBoost": XGBRegressor()
    }

    best_model = None
    best_score = 0

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        r2 = r2_score(y_test, preds)
        print(f"{name} R2 Score:", r2)

        if r2 > best_score:
            best_score = r2
            best_model = model

    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    print("Best Model Saved Successfully!")
