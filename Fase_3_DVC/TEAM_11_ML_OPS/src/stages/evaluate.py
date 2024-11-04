import argparse
import yaml
import pandas as pd
import joblib
from src.report.visualization import conf_matrix_plot
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def evaluate_metrics(config_path) -> None:
    config = yaml.safe_load(open(config_path))
    reg = joblib.load(config["train"]["model_path"]["grid_search_log_reg"])
    X_test= pd.read_csv(config["data"]["x_df_test_scaled"], delimiter=',')
    y_val= pd.read_csv(config["data"]["y_val_csv"], delimiter=',')
    X_test= pd.read_csv(config["data"]["x_df_test_scaled"], delimiter=',')
    predictions = reg.predict(X_test)
    conf_matrix = confusion_matrix(y_val, predictions)
    conf_matrix_plot(conf_matrix, reg)

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    evaluate_metrics(config_path=args.config)