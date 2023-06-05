from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report
from DataProcessing import *

#X_new = dimensionReduction(X)
X_new = filterDataset(X, liste)
X_line = filterDataset(X_test_test, liste)

X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2,random_state=0,stratify=y)
#take one line of X_train to test
X_train_line = X_train.iloc[10]
y_train_line = y_train.iloc[10]
print(y_train_line)
#transform to array
X_train_line = X_train_line.values
#change las line of X_line dataframe with x_train_line
print(X_line)
X_line.iloc[-1] = X_train_line
print(X_line)
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(X_train)
y_train_cluster = kmeans.predict(X_train)
y_test_cluster = kmeans.predict(X_test)
print(X_train.shape, X_test.shape)

""" Random Forest best parameter:  {'n_estimators': 1400, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 80, 'bootstrap': False} """
def run_randomForest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=1400, min_samples_split=5, min_samples_leaf=1, max_features='sqrt', max_depth=80, bootstrap=False, random_state=0)
    clf.fit(X_train, y_train)
    y_pred_rf = clf.predict(X_test)
    print('Accuracy on test set: ')
    #accuracy
    print('Accuracy: ', accuracy_score(y_test, y_pred_rf))
    #confusion matrix
    print('Confusion Matrix: ', confusion_matrix(y_test, y_pred_rf))
    #classification report
    print('Classification Report: ', classification_report(y_test, y_pred_rf))
    print(accuracy_score(y_test, y_pred_rf))
    return clf

#overfitting test
def overfitting_test(clf, X_train, X_test, y_train, y_test):
    """
    Fonction pour tester la présence de sur-ajustement (overfitting) d'un modèle.
    
    Args:
    clf : modèle de machine learning entraîné
    X_train : données d'entraînement (caractéristiques)
    X_test : données de test (caractéristiques)
    y_train : labels d'entraînement
    y_test : labels de test
    
    Returns:
    None (affiche les performances sur les ensembles d'entraînement et de test)
    """
    
    # Performance sur les ensembles d'entraînement
    train_accuracy = clf.score(X_train, y_train)
    train_predictions = clf.predict(X_train)
    train_precision = precision_score(y_train, train_predictions)
    train_recall = recall_score(y_train, train_predictions)
    train_f1 = f1_score(y_train, train_predictions)
    
    print("Performances sur l'ensemble d'entraînement :")
    print("Accuracy :", train_accuracy)
    print("Precision :", train_precision)
    print("Recall :", train_recall)
    print("F1-Score :", train_f1)
    
    # Performance sur les ensembles de test
    test_accuracy = clf.score(X_test, y_test)
    test_predictions = clf.predict(X_test)
    test_precision = precision_score(y_test, test_predictions)
    test_recall = recall_score(y_test, test_predictions)
    test_f1 = f1_score(y_test, test_predictions)
    
    print("\nPerformances sur l'ensemble de test :")
    print("Accuracy :", test_accuracy)
    print("Precision :", test_precision)
    print("Recall :", test_recall)
    print("F1-Score :", test_f1)


def predict_rf(clf, X_line):
    y_pred_rf = clf.predict(X_line)
    return y_pred_rf 
y_pred_km = kmeans.predict(X_line)
print(y_pred_km)
if __name__ == "__main__":
    clf = run_randomForest(X_train, X_test, y_train_cluster, y_test_cluster)

    y_pred_rf = predict_rf(clf, X_line)
    print(y_pred_rf)
    print("Done!") 
