# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 22:11:24 2023

@author: yannt
"""

# Import
#------------------------------------------------------------------------------
import warnings 
import pandas as pd
import numpy as np

import constant as C



from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, 
                                     cross_val_score, train_test_split)
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

                                     


#------------------------------------------------------------------------------
warnings.filterwarnings('ignore')
#------------------------------------------------------------------------------


def split_dataframe(data_path, target,to_replace, replace):
    df = pd.read_csv(data_path)
    df.dropna(inplace = True)
    X, y = df.drop([target], axis = 1), df[target]
    y.replace(to_replace, replace, inplace = True)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size = 0.2, 
                                                        stratify = y,
                                                        random_state = C.SEED)
    
    X_test, X_validate, y_test, y_validate = train_test_split(X_test, y_test,
                                                        test_size = 0.1, 
                                                        stratify = y_test,
                                                        random_state = C.SEED)
    return X_train, X_test, X_validate, y_train, y_test, y_validate




if __name__=='__main__':
    
    (X_train, X_test,X_validate, 
     y_train, y_test, y_validate) = split_dataframe(C.PATH_DATASET + C.SPAM, 
                                                       C.TARGET, 
                                                       ['benign', 'spam'], 
                                                       [0, 1])
                                                    
    """
    steps = [('scaler', StandardScaler()),
             ('knn', KNeighborsClassifier())]
    pipeline = Pipeline(steps = steps)
    
    parameters = {'knn__n_neighbors': np.arange(1, 50)}
    cv = GridSearchCV(pipeline, param_grid=parameters)
    cv.fit(X_train, y_train)
    y_pred = cv.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"acc : {acc}")
    """


    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
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
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    # Fit the random search model
    rf_random.fit(X_train, y_train)
    
    
   