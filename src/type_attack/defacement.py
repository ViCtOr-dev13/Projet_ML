#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 18:44:50 2023

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
scaler = StandardScaler()

df_defacement_best = pd.read_csv(C.PATH_DATASET + C.BEST_DEFACEMENT)
rf_defacement_best = RandomForestClassifier()
p.pre_preprocessing_pipeline(df_defacement_best, target_best, C.REPLACE_DEFACEMENT, C.REPLACE)

                                                  
(X_train_best, X_test_best, X_validate_best, 
 y_train_best, y_test_best, y_validate_best) = p.split_dataframe(df_defacement_best, 
                                                   target_best)                                                  

scaler.fit_transform(X_train_best)                                                             
rf_defacement_best.fit(X_train_best, y_train_best)

data = []
with open("../../datasets/URL/DefacementSitesURLFiltered.csv", 'r') as f:
    lines = f.readlines()
    for url in lines:
        data.append(to_csv.url_to_dico(url))
df = pd.DataFrame(data, columns=X_train_best.columns)
df.to_csv('../../result/prediction/defacement.csv', index=False)


df_1 = pd.read_csv('../../result/prediction/defacement.csv')
df_1 = df_1[X_train_best.columns]
scaler.transform(df_1)

a = rf_defacement_best.predict(df_1)
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

b = rf_defacement_best.predict(df_1)
print(b)

accuracy = (np.count_nonzero(a == 1)/len(a) + np.count_nonzero(b == 0)/len(b))/2



