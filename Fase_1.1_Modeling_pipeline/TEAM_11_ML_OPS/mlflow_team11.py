#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('team_11_ml_ops_module')  # Reemplaza 'team_11' con la ruta correcta de tu carpeta

from load_data import load_data
from explore_data import explore_data
from modeling.train import train_logistic_regression, train_random_forest, train_with_pca

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


# # Se leen los datos de divorcio

# In[2]:


datapath = 'DATA/divorce.csv'
data = load_data(datapath)


# # Se genera un mlflow con regresion logistica

# In[3]:


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(f"/team11/prediccion_divorsio")

with mlflow.start_run() as run:
    # Load the diabetes dataset.
    X = data.drop('Class', axis=1)
    y = data['Class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Explorar datos
    explore_data(data)

    # Log data split information
    mlflow.log_param("train_data_size", len(X_train))
    mlflow.log_param("test_data_size", len(X_test))

    # 2. Create a Pipeline for Preprocessing and Model Training
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=0.1, max_iter=1000, penalty='l1', solver='saga'))
    ])

    # 3. Train the Model
    pipeline.fit(X_train, y_train)

    # 4. Evaluate the Model
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log model performance metrics
    mlflow.log_metric("accuracy", accuracy)

    # Generate and print classification report
    report = classification_report(y_test, y_pred)
    print(report)
    mlflow.log_text(report, "classification_report.txt")  # Save report to artifacts

    # Generate and print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    mlflow.log_text(str(cm), "confusion_matrix.txt")  # Save matrix to artifacts

    # 5. Log Model with MLflow
    mlflow.sklearn.log_model(pipeline, "Divorce_Prediction_Model")

    # 6. Add any relevant tags or information
    mlflow.set_tag("developer", "Team 11")  # For example
    mlflow.set_tag("model_type", "LogisticRegression")

print("MLflow experiment completed. Check mlruns folder for results.")


# # Pipeline de todos los modelos
# 
#  - 1. Regresion logistica
#    2. Random Forest
#    3. PCA + Regresion logistica

# In[7]:


# Define una función para el entrenamiento y la evaluación
def run_experiment(model_name, model_func, X, y):
    with mlflow.start_run():
        mlflow.log_param("model_name", model_name)

        # Entrenamiento del modelo
        model, X_test, y_test = model_func(X, y)
        mlflow.sklearn.log_model(model, "model")  # Registra el modelo en MLflow

        # Evaluación del modelo
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)


        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        print(f"Modelo: {model_name}")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-score: {f1}")


# Preprocesamiento, entrenamiento y evaluación con MLflow

# Regresión Logística
run_experiment("Logistic Regression", train_logistic_regression, X, y)

# Random Forest
run_experiment("Random Forest", train_random_forest, X, y)

# PCA + Regresión Logística
run_experiment("PCA + Logistic Regression", train_with_pca, X, y)


# In[ ]:




