# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 00:03:25 2018

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
# Confirm the current working directory
os.getcwd()

# Import the dataset
dataset = pd.read_csv('Data.csv')

# Split the dataset into two matrix: independent, dependent variables matrix
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 3].values

# Show the created matrix in cosole
# Can't open it by variable explorer. It's not supported.
print(X)
print(y)

# Taking care of missing data
## Not good: remove the lines which contains missing value
## OK: Take a mean of the all value in the column
from sklearn.preprocessing import Imputer
# selecting the library or function name and press Ctrl+i will give you an tutorial of that function.
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])
print(X)























