import pandas as pd
import joblib
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

def train_logistic_regression(X, y, C, penalty, solver, max_iter):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(C=C, max_iter=max_iter, penalty=penalty, solver=solver)
    model.fit(X_train, y_train)
    return model, X_test, y_test

def train_random_forest(X, y, n_estimators, max_depth, min_samples_split, min_samples_leaf):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(random_state=42, n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    model.fit(X_train, y_train)
    return model, X_test, y_test

def train_with_pca(X, y, C, penalty, solver, max_iter):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    # Separar datos en entrenamiento y prueba
    X_train_pca, X_test_pca, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42
    )
    # Entrenar un clasificador binario
    model = LogisticRegression(C=C, max_iter=max_iter, penalty=penalty, solver=solver)
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
