# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 15:37:29 2023

@author: yannt
"""

# Import
#------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
import src.constant as C
import src.function.function as f


#------------------------------------------------------------------------------
warnings.filterwarnings('ignore')
#------------------------------------------------------------------------------
# Spam

df = pd.read_csv(C.PATH_DATASET + C.SPAM)
columns = f.get_nan_column(df)

for column in columns:
    print(f"{column} : {df[column].describe()}\n")

print(df[columns[0]].describe()['mean'])

f.preprocessing_target(df, C.TARGET, C.REPLACE_SPAM, C.REPLACE)
f.fill_nan_mean(df)

print(df.head(5))
    


#------------------------------------------------------------------------------

