# modeling/predict.py

import pandas as pd
import joblib
import sys

def load_model(model_path):
    """Carga el modelo desde el archivo especificado."""
    model = joblib.load(model_path)
    return model

def make_predictions(model, X_new_path):
    """Realiza predicciones utilizando el modelo cargado y los datos nuevos."""
    X_new = pd.read_csv(X_new_path)
    predictions = model.predict(X_new)
    return predictions

if __name__ == '__main__':
    model_path = sys.argv[1]  # Ruta del modelo a cargar
    X_new_path = sys.argv[2]   # Ruta de los nuevos datos para predecir

    model = load_model(model_path)
    predictions = make_predictions(model, X_new_path)

    # Imprimir las predicciones
    print("Predicciones:")
    print(predictions)
