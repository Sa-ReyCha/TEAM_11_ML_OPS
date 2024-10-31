from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def train_model_log_reg(X_df_training_scaled, X_df_validation_scaled, y_train, X_val, y_val, hyperparams):
    # Parametros para Log_Reg
    param_grid_LOG_REG = hyperparams["train"]["clf_params"]
    grid_search = GridSearchCV(LogisticRegression(), param_grid_LOG_REG, cv=5, scoring='precision')
    grid_search.fit(X_df_training_scaled, y_train)

    print("Mejores hiperparámetros encontrados:")
    print(grid_search.best_params_)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_val)

    conf_matrix = confusion_matrix(y_val, y_pred)
    conf_matrix_plot(conf_matrix,best_model)

    # Imprimir el reporte de clasificación
    class_report = classification_report(y_val, y_pred, zero_division=1)
    print("\nReporte de clasificación:")
    print(class_report)

    # BEST MODEL !!
    log_reg = LogisticRegression(C=0.1, max_iter=1000, penalty='l1', solver='saga')
    log_reg.fit(X_df_training_scaled, y_train)



    # Hacer predicciones con los datos de validación
    y_pred = log_reg.predict(X_df_validation_scaled)


    # Imprimir la matriz de confusión
    conf_matrix = confusion_matrix(y_val, y_pred, labels=log_reg.classes_)  # y_val es la verdadera etiqueta de validación
    conf_matrix_plot(conf_matrix,log_reg)
    # print(conf_matrix)

    # Imprimir el reporte de clasificación
    class_report = classification_report(y_val, y_pred)
    print("\nReporte de clasificación:")
    print(class_report)