# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 17:35:29 2023

@author: yannt
"""

# Import
#------------------------------------------------------------------------------
import warnings
import src.constant as C
import src.function.function as f

import pandas as pd


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                             classification_report, roc_curve)


# Formule
#------------------------------------------------------------------------------
# precision = (true positive/ (true positive + false positive))
# recall = true positive / (true positive + false negative)
# accuracy = (true positive + true negative) / (total)
# F1 score = 2 x (precision x recall)/ (precision + recall)

#------------------------------------------------------------------------------
warnings.filterwarnings('ignore')
#------------------------------------------------------------------------------

def print_accuracy_decision_tree(data_file, target):
    df = pd.read_csv(data_file)
    df = f.fill_nan_mean(df)
    (X_train, X_test,X_validate, 
     y_train, y_test, y_validate) = f.split_dataframe(data_file, 
                                                           target)
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Decision Tree accuracy : {acc}")
    
#------------------------------------------------------------------------------
# SPAM
spam = C.PATH_DATASET + C.SPAM
target = C.TARGET
print("spam")
print_accuracy_decision_tree(spam, target)

# defacmenet
defacmeent = C.PATH_DATASET + C.DEFACEMENT
target = C.TARGET
print("defacement")
print_accuracy_decision_tree(defacmeent, target)

# malware
malware = C.PATH_DATASET + C.MALWARE
target = C.TARGET
print("malware")
print_accuracy_decision_tree(malware, target)

# phishing
phishing = C.PATH_DATASET + C.PHISHING
target = C.TARGET
print("phishing")
print_accuracy_decision_tree(phishing, target)