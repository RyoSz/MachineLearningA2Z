# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 11:49:00 2018

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
os.chdir("C:\\Users\\Roy\\Desktop\\Projects\\Lectures\\MachineLearning\\Machine Learning A-Z\\Part 2 - Regression\Section 6 - Polynomial Regression")
# Confirm the current working directory
os.getcwd()

# Import the dataset
dataset = pd.read_csv('Position_Salaries.csv')


# Split the dataset into two matrix: independent, dependent variables matrix
## こうやって書くとMatrixとして認識され、
X = dataset.iloc[:, 1:2].values
## こうやって書くとVectorとして認識されるらしい。
y = dataset.iloc[:, 2].values

# Fitting the linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting the polynominal regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualizing the Linear Regression result
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('PositionLevel')
plt.ylabel('Salary')
plt.show()

# Visualizing the Polynomial Regression result
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('PositionLevel')
plt.ylabel('Salary')
plt.show()

# 次数を上げていけば行くほど、かなりフィットはする。　＞＞　なんでだっけ？

# Xの数を増やして、グラフをスムーズにする。
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('PositionLevel')
plt.ylabel('Salary')
plt.show()


# Predicting a new result with Linear model
lin_reg.predict(6.5)


# Predicting a new result with Polynomial model
lin_reg_2.predict(poly_reg.fit_transform(6.5))













