import argparse
import yaml
import pandas as pd

def data_load(config_path) -> None:
    config = yaml.safe_load(open(config_path))
    data = pd.read_csv(config["data"]["dataset_csv"], delimiter=';')
    print(config["data"]['dataset_processed_path'])
    data.to_csv(config["data"]['dataset_processed_path'], index=False)

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    data_load(config_path=args.config)