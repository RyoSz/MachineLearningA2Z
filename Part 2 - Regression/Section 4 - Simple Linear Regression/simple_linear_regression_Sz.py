# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 16:06:48 2018

@author: Roy
"""

# Data preprocessing
# Import libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Set working directory
# Needs to be changed path according to where file exists
os.chdir("C:\\Users\\sir1hig\\Desktop\\Projects\\Lectures\\MachineLearning\\MachineLearningA2Z\\Part 1 - Data Preprocessing\\Section 2 -------------------- Part 1 - Data Preprocessing --------------------")
os.chdir("C:\\Users\\Roy\\Desktop\\Projects\\Lectures\\MachineLearning\\Machine Learning A-Z\\Part 2 - Regression\Section 4 - Simple Linear Regression")
# Confirm the current working directory
os.getcwd()

# Import the dataset
dataset = pd.read_csv('Salary_Data.csv')

# Split the dataset into two matrix: independent, dependent variables matrix
# こうやって書くとMatrixとして認識され、
X = dataset.iloc[:, :-1].values
# こうやって書くとVectorとして認識されるらしい。
y = dataset.iloc[:, 1].values

#　Splitting dataset into train and test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state = 0)

# Fitting Simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set result
y_pred = regressor.predict(X_test)


# Visualizing the Training set result
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualizing the Test set result
plt.scatter(X_test, y_test, color='red')
## Regression lineは相変わらずTrain setのものを使う。
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()



















