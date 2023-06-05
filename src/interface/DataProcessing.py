import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

from from_url_to_csv import *

df_All = pd.read_csv('C:/Users/victo/OneDrive/Documents/CybersecurityProject/FinalDataset/All.csv')
df_All.head()
df_All["URL_Type_obf_Type"] = df_All["URL_Type_obf_Type"].replace(['benign'], 0).replace(['malware'], 1).replace(['phishing'], 2).replace(['spam'], 3).replace(['Defacement'], 4)
df_All.fillna(0, inplace=True)
df_All=df_All.replace([np.inf, -np.inf], np.nan)
df_All.dropna(inplace = True)
df_All.fillna(0, inplace=True)

X = df_All.drop('URL_Type_obf_Type', axis=1)
y = df_All['URL_Type_obf_Type']
url = "http://ap.org/Content/AP-In-The-News/2014/Former-AP-Latin-America-bureau-chief-dies"

data = parse_url(url)
df_test = pd.DataFrame([data], columns=['Querylength', 'domain_token_count', 'path_token_count',
           'avgdomaintokenlen', 'longdomaintokenlen', 'avgpathtokenlen', 'tld',
           'charcompvowels', 'charcompace', 'ldl_url', 'ldl_domain', 'ldl_path',
           'ldl_filename', 'ldl_getArg', 'dld_url', 'dld_domain', 'dld_path',
           'dld_filename', 'dld_getArg', 'urlLen', 'domainlength', 'pathLength',
           'subDirLen', 'fileNameLen', 'this.fileExtLen', 'ArgLen', 'pathurlRatio',
           'ArgUrlRatio', 'argDomanRatio', 'domainUrlRatio', 'pathDomainRatio',
           'argPathRatio', 'executable', 'isPortEighty', 'NumberofDotsinURL',
           'ISIpAddressInDomainName', 'CharacterContinuityRate',
           'LongestVariableValue', 'URL_DigitCount', 'host_DigitCount',
           'Directory_DigitCount', 'File_name_DigitCount', 'Extension_DigitCount',
           'Query_DigitCount', 'URL_Letter_Count', 'host_letter_count',
           'Directory_LetterCount', 'Filename_LetterCount',
           'Extension_LetterCount', 'Query_LetterCount', 'LongestPathTokenLength',
           'Domain_LongestWordLength', 'Path_LongestWordLength',
           'sub-Directory_LongestWordLength', 'Arguments_LongestWordLength',
           'URL_sensitiveWord', 'URLQueries_variable', 'spcharUrl',
           'delimeter_Domain', 'delimeter_path', 'delimeter_Count',
           'NumberRate_URL', 'NumberRate_Domain', 'NumberRate_DirectoryName',
           'NumberRate_FileName', 'NumberRate_Extension', 'NumberRate_AfterPath',
           'SymbolCount_URL', 'SymbolCount_Domain', 'SymbolCount_Directoryname',
           'SymbolCount_FileName', 'SymbolCount_Extension',
           'SymbolCount_Afterpath', 'Entropy_URL', 'Entropy_Domain',
           'Entropy_DirectoryName', 'Entropy_Filename', 'Entropy_Extension',
           'Entropy_Afterpath'])

X_test_test=df_test
#replace tld column by 0
X_test_test['tld'] = 0

def dimensionReduction(X):
    filter = VarianceThreshold(0.01)
    filter.fit(X)
    X_filter = filter.transform(X)
    
    #-------------------------------------
    #remove duplicated features
    X_T = X_filter.T
    

    #convert to dataframe
    X_T = pd.DataFrame(X_T)
    

    #remove duplicated
    X_T.duplicated().sum()
    duplicated_features = X_T.duplicated()
    features_to_keep = [not index for index in duplicated_features]

    X_unique = X_T[features_to_keep].T
    

    scaler = StandardScaler()
    scaler.fit(X_unique)

    X = scaler.transform(X_unique)
    

    #dataframe
    X = pd.DataFrame(X)
    
    #build correlation matrix
    corr_matrix = X.corr()
    def get_correlation(data, threshold):
        corr_col = set()
        corr_matrix = data.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i,j])> threshold:
                    colname = corr_matrix.columns[i]
                    corr_col.add(colname)
        return corr_col

    corr_features = get_correlation(X, 0.85)
    X.drop(labels=corr_features, axis=1, inplace=True)
    X = pd.DataFrame(X)
    return X

liste = [ 0,  1,  2,  3,  5,  6,  9, 11, 13, 14, 19, 21, 22, 24, 25, 30, 31,
            34, 35, 36, 41, 42, 46, 47, 48, 49, 50, 51, 52, 53, 54, 56, 57, 58,
            59, 60, 61, 67, 68, 69]
def filterDataset(X, liste):
    # Créer une liste des colonnes à conserver
    columns_to_keep = [col for col in X.columns if X.columns.get_loc(col) in liste]

    # Filtrer le DataFrame en utilisant la liste des colonnes à conserver
    X_filtered = X[columns_to_keep].copy()

    return X_filtered

    
if __name__ == "__main__":
   
    X_new = dimensionReduction(X)
    print(X_new.columns) 
    
    print(X_test_test.shape)
    X_test = filterDataset(X_test_test, liste)
    print(X_test.shape)
