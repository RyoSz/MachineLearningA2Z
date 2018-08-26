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
os.chdir("C:\\Users\\Roy\\Desktop\\Projects\\Lectures\\MachineLearning\\Machine Learning A-Z\\Part 1 - Data Preprocessing\\Section 2 -------------------- Part 1 - Data Preprocessing --------------------")
# Confirm the current working directory
os.getcwd()


# Import the dataset
dataset = pd.read_csv('Data.csv')


# Split the dataset into two matrix: independent, dependent variables matrix
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 3].values


# Show the created matrix in cosole
# Can't open it by variable explorer. It's not supported.
print(*X, sep='\n')
print(*y, sep='\n')


# Taking care of missing data
## Not good: remove the lines which contains missing value
## OK: Take a mean of the all value in the column
from sklearn.preprocessing import Imputer
# selecting the library or function name and press Ctrl+i will give you an tutorial of that function.
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])
print(*X, sep='\n')


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#　ここから3行のやり方では順序付きデータになってしまい、カントリーコードとしては適切でない。
#　しかしなぜかこれを通さないとデータタイプがobject >> float　に変わらず、次のDummy variableの作成に移れないので、一旦通す。
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
print(*X, sep='\n')
#　代わりにDummy variableを作る。
onehotencoder_X = OneHotEncoder(categorical_features=[0])
X = onehotencoder_X.fit_transform(X).toarray()
print(*X, sep='\n')
# y variableのPurchasedは、Yes, No　しかないので、0/1でOKなので、LabelEncoderでOK。
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
print(*y, sep='\n')


#　Splitting dataset into train and test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)













