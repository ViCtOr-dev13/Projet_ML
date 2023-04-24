# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 15:07:19 2023

@author: yannt
"""

# Import
#------------------------------------------------------------------------------
import warnings

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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



# Creation des dataframes
dataFrames = {}
for dataset_name in LISTE_ALL:
    name = dataset_name[:-4]
    dataFrames[name] = pd.read_csv(PATH_DATASET + dataset_name)
    

full_df = pd.concat(list(dataFrames.values()), ignore_index=True) 
full_df.dropna(inplace = True)
X, y = full_df.drop(TARGET, axis = 1), full_df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size = 0.2, 
                                                    random_state = SEED,
                                                    stratify = y)


rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(acc)


