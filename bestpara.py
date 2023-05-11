import pandas as pd 
import numpy as np

#scikit-learn KNN
from sklearn.neighbors import KNeighborsClassifier
#sckit learn random forest
from sklearn.ensemble import RandomForestClassifier
#sckit learn decision tree
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

global X_train, X_test, y_train, y_test
def malware():
    #import malware csv
    df_malware = pd.read_csv('FinalDataset/Malware.csv')
    df_malware["URL_Type_obf_Type"] = df_malware["URL_Type_obf_Type"].replace(['benign'], 1)
    df_malware["URL_Type_obf_Type"] = df_malware["URL_Type_obf_Type"].replace(['malware'], 0)
    #change NaN value by O
    #df_malware.fillna(0, inplace=True)
    #compter le nombre de ligne contenant NaN
    df_malware=df_malware.drop('NumberRate_Extension', axis=1)
    print(df_malware.isnull().sum().tail(10))
    df_malware.dropna(inplace = True)
    print(df_malware.info())
    # X and y
    X = df_malware.drop('URL_Type_obf_Type', axis=1)
    y = df_malware['URL_Type_obf_Type']

    return X, y

def defacement():
    #import Defacement csv
    df_Defacement = pd.read_csv('FinalDataset/Defacement.csv')
    df_Defacement["URL_Type_obf_Type"] = df_Defacement["URL_Type_obf_Type"].replace(['benign'], 1)
    df_Defacement["URL_Type_obf_Type"] = df_Defacement["URL_Type_obf_Type"].replace(['Defacement'], 0)
    #change NaN value by O
    #df_Defacement.fillna(0, inplace=True)
    #compter le nombre de ligne contenant NaN
    df_Defacement=df_Defacement.drop('Entropy_DirectoryName', axis=1)
    #drop la clonne NumberRate_Extension
    df_Defacement=df_Defacement.drop('NumberRate_Extension', axis=1)
    print(df_Defacement.isnull().sum().tail(10))
    df_Defacement.dropna(inplace = True)
    print(df_Defacement.info())
    # X and y
    X = df_Defacement.drop('URL_Type_obf_Type', axis=1)
    y = df_Defacement['URL_Type_obf_Type']
    return X, y



def knn(X,y):

    knn = KNeighborsClassifier()
     #RandomizedSearchCV
    weights=['uniform', 'distance']
    param_grid = {'n_neighbors': np.arange(1, 25),'weights':weights}
    knn_cv = RandomizedSearchCV(knn, param_grid, cv=5)
    return knn_cv


def randomforest(X,y):
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
    rf_cv = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    return rf_cv

def clf(X,y):

    clf = DecisionTreeClassifier()
 


    # n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    criterion = ['gini', 'entropy']
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
   
    # Minimum number of samples required to split a node
   
    
 
    random_grid = { 'criterion': criterion,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf}
    clf_cv = RandomizedSearchCV(clf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    return clf_cv

def main():
    #X, y = malware()
    X, y = defacement()
    knn_cv = knn(X,y)
    rf_cv = randomforest(X,y)
    clf_cv = clf(X,y)
    # fit the model with data
    knn_cv.fit(X, y)
    rf_cv.fit(X, y)
    clf_cv.fit(X, y)
    print("KNN best parameter: ", knn_cv.best_params_)
    print("Random Forest best parameter: ", rf_cv.best_params_)
    print("Decision Tree best parameter: ", clf_cv.best_params_)

if __name__ == "__main__":
    main()