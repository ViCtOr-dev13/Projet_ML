# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 18:52:27 2023

@author: yannt
"""


# Import
#------------------------------------------------------------------------------
import warnings

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
rf_spam = RandomForestClassifier(n_estimators= 1400,
                                 min_samples_split = 2,
                                 min_samples_leaf= 1,
                                 max_features = 'auto',
                                 max_depth = 40,
                                 bootstrap = False)
rf_spam.fit(X_train, y_train)
y_pred = rf_spam.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"accuracy : {acc}")
print(f"confusion matrix : {confusion_matrix(y_test, y_pred)}")
print(f"classification report  : {classification_report(y_test, y_pred)}")


y_pred_prob = rf_spam.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show()
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

rf_phishing = RandomForestClassifier()
rf_phishing.fit(X_train, y_train)
y_pred = rf_phishing.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"accuracy : {acc}")
print(f"confusion matrix : {confusion_matrix(y_test, y_pred)}")
print(f"classification report  : {classification_report(y_test, y_pred)}")


y_pred_prob = rf_phishing.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show()



