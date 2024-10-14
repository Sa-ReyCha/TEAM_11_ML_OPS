import pandas as pd
import joblib
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def train_logistic_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression(C=0.1, max_iter=1000, penalty='l1', solver='saga')
    model.fit(X_train, y_train)
    return model, X_test, y_test

def train_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    return model, X_test, y_test

def train_with_pca(X, y):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    # Separar datos en entrenamiento y prueba
    X_train_pca, X_test_pca, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42
    )
    # Entrenar un clasificador binario
    model = LogisticRegression()
    model.fit(X_pca, y)
    return model, X_test_pca, y_test
    
if __name__ == '__main__':
    model_choice = sys.argv[1]  # Elige el modelo: 'logistic', 'random_forest' o 'pca'
    X_train_path = sys.argv[2]
    y_train_path = sys.argv[3]
    
    if model_choice == 'logistic':
        model = train_logistic_regression(X_train_path, y_train_path)
        joblib.dump(model, 'logistic_model.pkl')
    elif model_choice == 'random_forest':
        model = train_random_forest(X_train_path, y_train_path)
        joblib.dump(model, 'random_forest_model.pkl')
    elif model_choice == 'pca':
        train_with_pca(X_train_path, y_train_path)
    else:
        print("Modelo no reconocido. Por favor elige 'logistic', 'random_forest' o 'pca'.")
