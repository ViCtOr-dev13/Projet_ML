# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 16:34:58 2023

@author: yannt
"""

# Import
#------------------------------------------------------------------------------
import warnings

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                             classification_report, roc_curve)


# Formule
#------------------------------------------------------------------------------
# precision = (true positive/ (true positive + false positive))
# recall = true positive / (true positive + false negative)
# accuracy = (true positive + true negative) / (total)
# F1 score = 2 x (precision x recall)/ (precision + recall)

# Constant
#------------------------------------------------------------------------------
PATH_DATASET = "../datasets/"
ALL = "All.csv"
DEFACEMENT = "Defacement.csv"
MALWARE = "Malware.csv"
PHISHING = "Phishing.csv"
SPAM = "Spam.csv"
LISTE_ALL = [DEFACEMENT, MALWARE, PHISHING, SPAM]
TARGET = "URL_Type_obf_Type"

SEED = 1

# useful code
#------------------------------------------------------------------------------
warnings.filterwarnings('ignore')
#------------------------------------------------------------------------------

# SPAM
df_spam = pd.read_csv(PATH_DATASET + SPAM)
df_spam.dropna(inplace = True)
X, y = df_spam.drop([TARGET], axis = 1), df_spam[TARGET]
y.replace(['benign', 'spam'], 
          [0, 1], 
          inplace= True)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                     test_size = 0.2,
                                     stratify = y,
                                     random_state = SEED)

lr = LogisticRegression()
knn = KNN()
dt = DecisionTreeClassifier()

classifiers = {('Logisitic Regression', lr),
               ('K Nearest Neighbours', knn),
               ('Classification Tree', dt)}

for clf_name, clf in classifiers:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"{clf_name} : {accuracy_score(y_test, y_pred)}")


vc = VotingClassifier(estimators = classifiers)
vc.fit(X_train, y_train)
y_pred = vc.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"acc : {acc}")




#------------------------------------------------------------------------------
# PHISING

df_phishing = pd.read_csv(PATH_DATASET + PHISHING)
df_phishing.dropna(inplace = True)
X, y = df_phishing.drop([TARGET], axis = 1), df_phishing[TARGET]
y.replace(['benign', 'phishing'], 
          [0, 1], 
          inplace= True)


X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                     test_size = 0.2,
                                     stratify = y,
                                     random_state = SEED)

lr = LogisticRegression()
knn = KNN()
dt = DecisionTreeClassifier()

classifiers = {('Logisitic Regression', lr),
               ('K Nearest Neighbours', knn),
               ('Classification Tree', dt)}

for clf_name, clf in classifiers:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"{clf_name} : {accuracy_score(y_test, y_pred)}")


vc = VotingClassifier(estimators = classifiers)
vc.fit(X_train, y_train)
y_pred = vc.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"acc : {acc}")


