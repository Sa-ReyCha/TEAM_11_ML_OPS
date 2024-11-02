import argparse
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split


def data_split(data):
    X = data.drop('Class', axis=1).copy()
    y = data['Class'].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    return X_train, X_test, y_train, y_test, X_val, y_val

def data_split_stage(config_path) -> None:
    config = yaml.safe_load(open(config_path))
    data = pd.read_csv(config["data"]["dataset_csv"], delimiter=';')
    X_train, X_test, y_train, y_test, X_val, y_val = data_split(data)
    X_train.to_csv(config["data"]["x_train_csv"], index=False)
    X_test.to_csv(config["data"]["x_test_csv"], index=False)
    y_train.to_csv(config["data"]["y_train_csv"], index=False)
    y_test.to_csv(config["data"]["y_test_csv"], index=False)
    X_val.to_csv(config["data"]["x_val_csv"], index=False)
    y_val.to_csv(config["data"]["y_val_csv"], index=False)

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    data_split_stage(config_path=args.config)