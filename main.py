import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
from sklearn.model_selection import train_test_split
import plotly.express as px 


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
    print(recipe.head())
    return recipe

download_data()

    