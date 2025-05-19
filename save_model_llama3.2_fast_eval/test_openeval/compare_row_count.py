import pandas as pd

df = pd.read_csv('gold.csv')
df_pred = pd.read_csv('predictions.csv')

print(f'gold: {len(df)}')
print(f'pred: {len(df_pred)}')

difference = len(df) - len(df_pred)
print(f'difference: {difference}')