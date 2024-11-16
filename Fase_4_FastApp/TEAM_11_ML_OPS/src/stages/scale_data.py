import argparse
import yaml
import pandas as pd
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

def data_scale(config_path) -> None:
    config = yaml.safe_load(open(config_path))
    X_train = pd.read_csv(config["data"]["x_train_csv"], delimiter=',')
    X_val = pd.read_csv(config["data"]["x_val_csv"], delimiter=',')
    X_test = pd.read_csv(config["data"]["x_test_csv"], delimiter=',')

    X_df_training_scaled, X_df_validation_scaled, X_df_test_scaled = scale_data(X_train, X_val, X_test)

    pd.DataFrame(X_df_training_scaled).to_csv(config["data"]['x_df_training_scaled'], index=False)
    pd.DataFrame(X_df_validation_scaled).to_csv(config["data"]['x_df_validation_scaled'], index=False)
    pd.DataFrame(X_df_test_scaled).to_csv(config["data"]['x_df_test_scaled'], index=False)

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    data_scale(config_path=args.config)