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



