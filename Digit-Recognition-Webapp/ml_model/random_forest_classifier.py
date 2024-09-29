import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import joblib

try:
    data = pd.read_csv('./data/train.csv')
    print("Successfully loaded data 👍.")
except:
    print("Error 438! While loading data.")
# Splitting data for features and targeted columns.
try:
    x = data.drop('label', axis=1)
    y = data['label']
    print("Data is successfully splitted into x & y 👍.")
except:
    print('Error 745! While spitting data into x & y')
# Splitting data into training and testing.
try:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    print("Successfully splitted into training & testing 👍.")
except:
    print("Erroe while splitting data into training and testing!")
# Load out model.
model = RandomForestClassifier()
try:
    print("Model is learning from our data.....")
    model.fit(x_train, y_train)
    print("Model successfully trained 👍.")
except:
    print("Error while training!")
# Predicting using testing.
try:
    print("Predicting......")
    y_predicted = model.predict(x_test)
    print("Model successuly tested!")
except:
    print("An error occured while testing!")
# Calculating accuracy score.
try:
    score = accuracy_score(y_test, y_predicted)
    print(f'Our model is {np.round((score*100), 2)}% accurate!')
except:
    print("An error occured when calculating accuracy score!")
# Dumbing data into our file.
try:
    joblib.dump(model, 'decision_tree_model.joblib')
    print("Data Successfully Dumbed 👍.")
except:
    print("An error while dumbing data!")

def prediction(arr):
    return model.predict(arr)