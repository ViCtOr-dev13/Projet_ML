#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 18:44:34 2023

@author: yannt
"""

# Import
#------------------------------------------------------------------------------
import warnings
import pandas as pd
import numpy as np
import json

import src.constant as C
import src.function.preprocessing as p
import src.convert_url_to_csv as to_csv


from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

#------------------------------------------------------------------------------
warnings.filterwarnings('ignore')
#------------------------------------------------------------------------------

target_best = C.TARGET_BEST


df_best = pd.read_csv(C.PATH_DATASET + C.BEST_SPAM)
rf_best = RandomForestClassifier()

p.pre_preprocessing_pipeline(df_best, target_best, C.REPLACE_SPAM, C.REPLACE)

                                                  
(X_train_best, X_test_best, X_validate_best, 
 y_train_best, y_test_best, y_validate_best) = p.split_dataframe(df_best, 
                                                   target_best)      
                                                                 

scaler = StandardScaler()

                                                                      

scaler.fit_transform(X_train_best)                                                             
rf_best.fit(X_train_best, y_train_best)

data = []
with open("../../datasets/URL/spam_dataset.csv", 'r') as f:
    lines = f.readlines()
    for url in lines:
        data.append(to_csv.url_to_dico(url))
df = pd.DataFrame(data, columns=X_train_best.columns)
df.to_csv('../../result/prediction/spam.csv', index=False)


df_1 = pd.read_csv('../../result/prediction/spam.csv')
df_1 = df_1[X_train_best.columns]
scaler.transform(df_1)

a = rf_best.predict(df_1)
print(a)

data = []
with open("../../datasets/URL/Benign_list_big_final.csv", 'r') as f:
    lines = f.readlines()
    for url in lines:
        data.append(to_csv.url_to_dico(url))
df = pd.DataFrame(data, columns=X_train_best.columns)
df.to_csv('../../result/prediction/benign.csv', index=False)


df_1 = pd.read_csv('../../result/prediction/benign.csv')
df_1 = df_1[X_train_best.columns]
scaler.transform(df_1)

b = rf_best.predict(df_1)
print(b)

                                                                 

