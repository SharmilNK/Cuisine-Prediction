# Cuisine Prediction Using Machine Learning

# Overview

This project predicts the cuisine category (e.g., Indian, Italian, Chinese, Greek) of a dish based on its list of ingredients.
It uses machine learning models — Logistic Regression and Random Forest — to classify cuisines using text-based ingredient data from the popular Kaggle Recipe Ingredients Dataset.

# Problem Statement
Given a list of ingredients, we aim to determine which cuisine a dish belongs to.


# Dataset
The dataset is sourced from Kaggle:
Recipe Ingredients Dataset


# Project Structure
Cuisine-Prediction/
│
├── data/
│   ├── train.json
│   └── test.json
│
├── main.py
└── README.md

# Machine Learning Pipeline

# Data Loading
Load dataset from local .json files.

# Preprocessing
Convert list of ingredients into a single text string.

# Feature Extraction
Use TF-IDF Vectorization to transform text into numerical features.

Model Training : Logistic Regression

# Prediction
Load the saved model to predict cuisine for user-input ingredients.

# Dependencies
Install the following Python libraries before running:

pip install numpy pandas matplotlib seaborn plotly scikit-learn joblib

# How to Run
Clone the repository

git clone https://github.com/SharmilNK/Cuisine-Prediction.git
cd Cuisine-Prediction

# Place the dataset
Download train.json and test.json from Kaggle and place them in the data/ folder.

# Run the program
python main.py