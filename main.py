import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
from sklearn.model_selection import train_test_split
import plotly.express as px
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib


def download_data():
    '''
    Download the recipe-ingredients-dataset data set from Kaggle
    Output: recipe - df containing data 
    '''
    import os
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))

    recipe = pd.read_json(r'C:\Users\som\Desktop\CourseSem1\Sourcing\Cuisine-Prediction\train.json')
    test = pd.read_json(r'C:\Users\som\Desktop\CourseSem1\Sourcing\Cuisine-Prediction\test.json')
    print(recipe.head())
    return df


#  Train the model
def train_model(df):
    X = df['text']
    y = df['cuisine']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build pipeline
    model = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression(max_iter=200))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(model, 'cuisine_model.pkl')
    print(" Model saved as cuisine_model.pkl")

# Predict cuisine for user input
def predict_cuisine(ingredients_list):
    model = joblib.load('cuisine_model.pkl')
    text = ' '.join(ingredients_list)
    pred = model.predict([text])[0]
    print(f"Predicted cuisine: {pred}")

# Example run
if __name__ == "__main__":
    df = download_data
    train_model(df)

    # Example user input
    user_ingredients = ["soy sauce", "ginger", "garlic", "rice", "chicken"]
    predict_cuisine(user_ingredients)


    