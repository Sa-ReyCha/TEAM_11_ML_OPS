import pandas as pd
import joblib
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def train_logistic_regression(X_train_path, y_train_path):
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)
    
    model = LogisticRegression(C=0.1, max_iter=1000, penalty='l1', solver='saga')
    model.fit(X_train, y_train.values.ravel())
    return model

def train_random_forest(X_train_path, y_train_path):
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)

    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train.values.ravel())
    return model

def train_with_pca(X_path, y_path):
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Dividir datos en entrenamiento y prueba
    from sklearn.model_selection import train_test_split
    X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    # Entrenar un clasificador binario
    model = LogisticRegression()
    model.fit(X_train_pca, y_train)

    # Hacer predicciones
    y_pred = model.predict(X_test_pca)

    # Calcular precisión
    accuracy = accuracy_score(y_test, y_pred)
    print("Precisión del modelo con PCA:", accuracy)

    # Graficar las predicciones
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_pred, cmap="viridis", alpha=0.7, edgecolors="k")
    plt.xlabel("Componente principal 1")
    plt.ylabel("Componente principal 2")
    plt.title("Predicciones del clasificador binario con PCA")
    plt.colorbar(label="Predicción")
    plt.show()

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
