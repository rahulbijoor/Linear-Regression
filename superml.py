#!/usr/bin/env python
# coding: utf-8

#Linear Regression is a supervised machine learning techinique used to predict a numeric value from one or more input features. 

# This program uses linear regression to predict the scores of students based on the number of hours they studied.

# first we import the necessary libraries
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



# we get the data from file student_scores.csv
df = pd.read_csv('student_scores.csv')
print(df.head(10))

X=df['Hours']  # X is the value to be predicted
Y=df['Scores'] # Y is the value based on which the prediction is made

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=50)

# Create a Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)