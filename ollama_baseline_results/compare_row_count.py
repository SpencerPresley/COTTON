import pandas as pd

HUMANEVAL_GOLD_PATH = '../save_model_llama3.2_fast_eval/test_humaneval/gold.csv'
OPENEVAL_GOLD_PATH = '../save_model_llama3.2_fast_eval/test_openeval/gold.csv'

HUMAN_EVAL_PRED_PATH = './test_humaneval/predictions.csv'
OPENEVAL_PRED_PATH = './test_openeval/predictions.csv'

df_humaneval_gold = pd.read_csv(HUMANEVAL_GOLD_PATH)
df_humaneval_pred = pd.read_csv(HUMAN_EVAL_PRED_PATH)
df_openeval_gold = pd.read_csv(OPENEVAL_GOLD_PATH)
df_openeval_pred = pd.read_csv(OPENEVAL_PRED_PATH)

print(f'humaneval gold: {len(df_humaneval_gold)}')
print(f'humaneval pred: {len(df_humaneval_pred)}')
print(f'openeval gold: {len(df_openeval_gold)}')
print(f'openeval pred: {len(df_openeval_pred)}')

print(f"humaneval vs gold difference: {len(df_humaneval_gold) - len(df_humaneval_pred)}")
print(f"openeval vs gold difference: {len(df_openeval_gold) - len(df_openeval_pred)}")



