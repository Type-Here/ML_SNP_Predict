from src.data_load import load_data

df = load_data(protein = 'p53')
print(df.head())