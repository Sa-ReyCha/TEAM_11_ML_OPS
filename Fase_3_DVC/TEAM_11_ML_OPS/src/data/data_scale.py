from sklearn.preprocessing import StandardScaler

def scale_data(X_train, X_val, X_test):
    # STD SCALER
    scaler = StandardScaler()


    # Ajustamos y transformamos los datos de entrenamiento
    X_df_training_scaled = scaler.fit_transform(X_train)

    # Transformamos los datos de Validacion
    X_df_validation_scaled = scaler.transform(X_val)
    # Transformamos los datos de Validacion
    X_df_test_scaled = scaler.transform(X_test)
    
    return X_df_training_scaled, X_df_validation_scaled, X_df_test_scaled