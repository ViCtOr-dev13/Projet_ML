# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 19:11:33 2023

@author: yannt
"""

# Import
#------------------------------------------------------------------------------
import src.constant as C
import warnings
import src.function.function as f


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                             classification_report, roc_curve)

#------------------------------------------------------------------------------
warnings.filterwarnings('ignore')
#------------------------------------------------------------------------------


if __name__=="__main__":
    
    df = pd.read_csv(C.PATH_DATASET + C.SPAM)
    f.drop_na_target(df, C.TARGET)
    f.replace_target(df, C.TARGET, C.REPLACE_SPAM, C.REPLACE)
    f.fill_nan_mean(df)
    (X_train, X_test,X_validate, 
     y_train, y_test, y_validate) = f.split_dataframe(df, C.TARGET)
    
    
                                            
    rf = RandomForestClassifier(n_estimators= 1400,
                                     min_samples_split = 2,
                                     min_samples_leaf= 1,
                                     max_features = 'auto',
                                     max_depth = 40,
                                     bootstrap = False)
    steps = [('scaler', StandardScaler()),
             ('random forest', rf)]
    

    pipeline = Pipeline(steps)
    
    rf_scaled = pipeline.fit(X_train, y_train)
    
    y_pred = rf_scaled.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"accuracy : {acc}")
    print(f"confusion matrix : {confusion_matrix(y_test, y_pred)}")
    print(f"classification report  : {classification_report(y_test, y_pred)}")




