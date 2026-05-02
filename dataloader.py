import pandas as pd

def load_data(path):
    data = pd.read_csv(path, sep=',', header='infer')
            
    return data