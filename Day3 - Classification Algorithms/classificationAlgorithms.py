import pandas as pd

file_path = "/Users/fsa/Desktop/data/irisDataset/IRIS.csv"
df = pd.read_csv(file_path)
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())