import pandas as pd
import sys

def explore_data(data):
    print(data)
    print(data.Class.value_counts())
    print(data.info())
    print(data.describe())
    print("Verificando valores nulos en el conjunto de datos:")
    print(data.isnull().sum())
    data = data.dropna()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
        data = pd.read_csv(data_file)
        explore_data(data)
    else:
        print("Uso: python script.py data.csv")