#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 18:44:45 2023

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

df = pd.read_csv(C.PATH_DATASET + C.PHISHING)
rf = RandomForestClassifier()
p.pre_preprocessing_pipeline(df, target, C.REPLACE_PHISHING, C.REPLACE)




df_best = pd.read_csv(C.PATH_DATASET + C.BEST_PHISHING)
rf_best = RandomForestClassifier()
p.pre_preprocessing_pipeline(df, target_best, C.REPLACE_PHISHING, C.REPLACE)




(X_train, X_test,X_validate, 
 y_train, y_test, y_validate) = p.split_dataframe(df, 
                                                   target)
                                                  
(X_train_best, X_test_best, X_validate_best, 
 y_train_best, y_test_best, y_validate_best) = p.split_dataframe(df_best, 
                                                   target_best)                                                  


