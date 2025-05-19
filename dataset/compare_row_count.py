import pandas as pd

df_cot = pd.read_csv('CoT_9k_cons.csv')
df_org_humaneval = pd.read_csv('humaneval.csv')
df_org_openeval = pd.read_csv('openeval.csv')

df_gold_humaneval = pd.read_csv('../save_model_llama3.2_fast_eval/test_humaneval/gold.csv')
df_gold_openeval = pd.read_csv('../save_model_llama3.2_fast_eval/test_openeval/gold.csv')

df_pred_ft_humaneval = pd.read_csv('../save_model_llama3.2_fast_eval/test_humaneval/predictions.csv')
df_pred_ft_openeval = pd.read_csv('../save_model_llama3.2_fast_eval/test_openeval/predictions.csv')

df_pred_ollamabase_humaneval = pd.read_csv('../ollama_baseline_results/test_humaneval/predictions.csv')
df_pred_ollamabase_openeval = pd.read_csv('../ollama_baseline_results/test_openeval/predictions.csv')

print(f'CoT_9k_cons.csv row count: {len(df_cot)}')
print(f'Original humaneval.csv row count: {len(df_org_humaneval)}')
print(f'Original openeval.csv row count: {len(df_org_openeval)}')
print(f'HumanEval gold.csv row count: {len(df_gold_humaneval)}')
print(f'OpenEval gold.csv row count: {len(df_gold_openeval)}')
print(f'Ollama base (no finetuning) humaneval predictions.csv row count: {len(df_pred_ollamabase_humaneval)}')
print(f'Ollama base (no finetuning) openeval predictions.csv row count: {len(df_pred_ollamabase_openeval)}')
print(f'Finetuned humaneval predictions.csv row count: {len(df_pred_ft_humaneval)}')
print(f'Finetuned openeval predictions.csv row count: {len(df_pred_ft_openeval)}')

print(f"NOTE: The original humaneval.csv, openeval.csv, and CoT_9k_cons.csv \nhave header rows which may be being counted, thus you may want to subtract 1 \nfrom their row counts above")


