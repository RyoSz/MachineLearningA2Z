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

# Set working directory
"""
- Go to file directory by file explorer
- Click on the right upper side botton as "set as current console's working directory"
"""

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























