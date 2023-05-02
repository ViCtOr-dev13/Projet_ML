# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 22:11:24 2023

@author: yannt
"""

# Import
#------------------------------------------------------------------------------
import warnings 
import numpy as np
import pandas as pd
import json

import time

import src.constant as C
import src.function.function as f

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier


#------------------------------------------------------------------------------
warnings.filterwarnings('ignore')
#------------------------------------------------------------------------------

if __name__=='__main__':
    df_spam = pd.read_csv(C.PATH_DATASET + C.SPAM)
    f.drop_na_target(df_spam, C.TARGET)
    f.replace_target(df_spam, C.TARGET, C.REPLACE_SPAM, C.REPLACE)
    f.fill_nan_mean(df_spam)

    (X_train, X_test,X_validate, 
     y_train, y_test, y_validate) = f.split_dataframe(df_spam, 
                                                       C.TARGET)


    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 200)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 200, num = 20)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    
        # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores

    
    rf_random = RandomizedSearchCV(estimator = rf,
                                   param_distributions = random_grid, 
                                   n_iter = 50 , cv = 3,
                                   random_state=42, n_jobs = -1)
    # Fit the random search model
    

    rf_random.fit(X_train, y_train)

    
    output = {}
    output['best_params'] = rf_random.best_params_
    output['best_estimator'] = str(rf_random.best_estimator_)
    output['best_score'] = rf_random.best_score_
    

    with open('../../result/randomizedsearch.json', 'w', encoding='utf8') as outfile:
        json.dump(output, outfile, indent = 4, ensure_ascii=False)
