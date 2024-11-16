import pandas as pd

def load_data(filepath):
    data = pd.read_csv(filepath, delimiter=';')
    return data