#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 20:08:44 2023

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


from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier


#------------------------------------------------------------------------------
warnings.filterwarnings('ignore')
#------------------------------------------------------------------------------

target = C.TARGET
target_best = C.TARGET_BEST

df_all = pd.read_csv(C.PATH_DATASET + C.ALL)
rf_all = RandomForestClassifier()
p.pre_preprocessing_pipeline(df_all, target, C.REPLACE_ALL, C.REPLACE_1)



df_all_best = pd.read_csv(C.PATH_DATASET + C.BEST_ALL)
rf_all_best = RandomForestClassifier()
p.pre_preprocessing_pipeline(df_all_best, target_best, C.REPLACE_ALL, C.REPLACE_1)




(X_train, X_test,X_validate, 
 y_train, y_test, y_validate) = p.split_dataframe(df_all, 
                                                   target)
                                                  
(X_train_best, X_test_best, X_validate_best, 
 y_train_best, y_test_best, y_validate_best) = p.split_dataframe(df_all_best, 
                                                   target_best)                                                  



