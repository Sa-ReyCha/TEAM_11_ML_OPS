import argparse
import yaml
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.report.visualization import viz_data_bar_graph, conf_matrix_plot
import joblib

def train_model_log_reg(X_df_training_scaled, X_df_validation_scaled, y_train, X_val, y_val, hyperparams):
    # Parametros para Log_Reg
    param_grid_LOG_REG = hyperparams["train"]["clf_params"]
    grid_search = GridSearchCV(LogisticRegression(), param_grid_LOG_REG, cv=5, scoring='precision')
    grid_search.fit(X_df_training_scaled, y_train)
    

    #print("Mejores hiperparámetros encontrados:")
    #print(grid_search.best_params_)

    best_model_log_reg = grid_search.best_estimator_
    joblib.dump(best_model_log_reg, hyperparams["train"]["model_path"]["grid_search_log_reg"])
    #y_pred = best_model.predict(X_val)

    #conf_matrix = confusion_matrix(y_val, y_pred)
    #conf_matrix_plot(conf_matrix,best_model)

    # Imprimir el reporte de clasificación
    #class_report = classification_report(y_val, y_pred, zero_division=1)
    #print("\nReporte de clasificación:")
    #print(class_report)

    # BEST MODEL !!
    log_reg = LogisticRegression(C=0.1, max_iter=1000, penalty='l1', solver='saga')
    log_reg.fit(X_df_training_scaled, y_train)
    joblib.dump(log_reg, hyperparams["train"]["model_path"]["log_reg"])


    # Hacer predicciones con los datos de validación
    #y_pred = log_reg.predict(X_df_validation_scaled)


    # Imprimir la matriz de confusión
    #conf_matrix = confusion_matrix(y_val, y_pred, labels=log_reg.classes_)  # y_val es la verdadera etiqueta de validación
    #conf_matrix_plot(conf_matrix,log_reg)
    # print(conf_matrix)

    # Imprimir el reporte de clasificación
    #class_report = classification_report(y_val, y_pred)
    #print("\nReporte de clasificación:")
    #print(class_report)

def train_model(config_path) -> None:
    config = yaml.safe_load(open(config_path))
    X_df_training_scaled = pd.read_csv(config["data"]["x_df_training_scaled"], delimiter=',')
    X_df_validation_scaled = pd.read_csv(config["data"]["x_df_training_scaled"], delimiter=',')
    y_train = pd.read_csv(config["data"]["y_train_csv"], delimiter=',')
    X_val = pd.read_csv(config["data"]["x_val_csv"], delimiter=',')
    y_val = pd.read_csv(config["data"]["y_val_csv"], delimiter=',')
    train_model_log_reg(X_df_training_scaled, X_df_validation_scaled, y_train, X_val, y_val, config)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    train_model(config_path=args.config)