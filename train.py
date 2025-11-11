import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib


def download_data():
    """
    Load the recipe-ingredients-dataset data set (from local train.json)
    Output: recipe - DataFrame containing data 
    """
    recipe = pd.read_json('train.json')
    print("✅ Data loaded successfully!")
    print(recipe.head())
    return recipe


def train_model(df):
    """
    Train a cuisine prediction model based on ingredients text.
    """
    # Combine ingredient list into text
    df['text'] = df['ingredients'].apply(lambda x: ' '.join(x))
    X = df['text']
    y = df['cuisine']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build pipeline
    model = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression(max_iter=200))
    ])

    # Train
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate
    print(f"\n🎯 Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(model, 'cuisine_model.pkl')
    print("\n💾 Model saved as cuisine_model.pkl")


def predict_cuisine(ingredients_list):
    """
    Load the trained model and predict cuisine type from a list of ingredients.
    """
    model = joblib.load('cuisine_model.pkl')
    text = ' '.join(ingredients_list)
    pred = model.predict([text])[0]
    print(f"🍽️ Predicted cuisine: {pred}")


if __name__ == "__main__":
    df = download_data()
    train_model(df)

    # Example user input
    user_ingredients = ["soy sauce", "ginger", "garlic", "rice", "chicken"]
    predict_cuisine(user_ingredients)
