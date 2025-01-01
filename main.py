from src.data_prep import load_data

df = load_data(protein = 'p53')
print(df.head())