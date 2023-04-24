# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 16:20:06 2023

@author: yannt
"""

# Import 
#------------------------------------------------------------------------------
import constant as C

import pandas as pd
import warnings


from sklearn.model_selection import train_test_split
from sklearn.ensemble import (VotingClassifier, BaggingClassifier,
                              RandomForestClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                             classification_report, roc_curve)

#------------------------------------------------------------------------------
warnings.filterwarnings("ignore")
#------------------------------------------------------------------------------
# Function
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

                                                
def accuracy_classifiers(classifiers, X_train, y_train, X_validate, y_validate):
    accuracy_liste = []
    for clf_name, clf in classifiers:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_validate)
        acc = accuracy_score(y_validate, y_pred)
        accuracy_liste.append([clf_name, acc])
    return accuracy_liste


#------------------------------------------------------------------------------

if __name__=="__main__":
    
    (X_train, X_test,X_validate, 
     y_train, y_test, y_validate) = split_dataframe(C.PATH_DATASET + C.SPAM, 
                                                       C.TARGET, 
                                                       ['benign', 'spam'], 
                                                       [0, 1])
                                            
    lr = LogisticRegression()
    knn = KNN()
    dt = DecisionTreeClassifier()
    estimators = {('Logisitic Regression', lr),
                  ('Decision Tree', dt),
                  ('K Nearest Neighbours', knn)}
    
    vc = VotingClassifier(estimators = estimators)
    bg = BaggingClassifier()
    rf = RandomForestClassifier(n_estimators = 400,
                               min_samples_leaf = 0.12,
                               random_state = C.SEED)
    
    classifiers = {('Logisitic Regression', lr),
                   ('K Nearest Neighbours', knn),
                   ('Classification Tree', dt),
                   ('Voting Classifiers', vc), 
                   ('Bagging', bg), 
                   ('Random Forest', rf)}
    
    accuracy_liste = accuracy_classifiers(classifiers, 
                                          X_train, y_train, X_validate, y_validate)
    
    for clf_name, accuracy_score in accuracy_liste:
        print(f"{clf_name} : {accuracy_score}")
        
        
        
        
