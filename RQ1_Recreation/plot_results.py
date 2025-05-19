import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import json # Added for loading nlgeval scores
import numpy as np # Added for grouped bar chart

# Default path for the summary CSV, relative to this script's location
DEFAULT_CSV_PATH = os.path.join(os.path.dirname(__file__), "pass_at_1_summary.csv")
DEFAULT_PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")

def plot_overall_performance_by_model(df: pd.DataFrame, output_dir: str):
    """Plots average pass@1 by model, averaged over datasets and plan types."""
    if df.empty:
        print("Skipping overall performance plot: DataFrame is empty.")
        return

    plt.figure(figsize=(24, 12)) # Slightly larger figure
    overall_performance = df.groupby('model_name')['pass_at_1'].mean().sort_values(ascending=False)
    
    # Address FutureWarning by assigning hue and disabling legend when palette is used for single series bar colors
    sns.barplot(x=overall_performance.index, y=overall_performance.values, \
                hue=overall_performance.index, palette="viridis", legend=False) # Added hue and legend=False
    
    plt.title('Overall Average Pass@1 by Model\n(Averaged across Datasets and Plan Types)', fontsize=12)
    plt.xlabel('Model Name', fontsize=12)
    plt.ylabel('Average Pass@1 Score (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout() # Apply tight_layout to adjust spacings
    
    plot_path = os.path.join(output_dir, "overall_performance_by_model.png")
    plt.savefig(plot_path)
    print(f"Saved plot: {plot_path}")
    plt.close()

def plot_performance_by_model_and_dataset(df: pd.DataFrame, output_dir: str):
    """Plots pass@1 by model, broken down by dataset (averaged over plan types)."""
    if df.empty:
        print("Skipping performance by model and dataset plot: DataFrame is empty.")
        return

    plt.figure(figsize=(15, 10)) # Slightly larger figure
    agg_df = df.groupby(['model_name', 'dataset'])['pass_at_1'].mean().reset_index()
    
    sns.barplot(x='model_name', y='pass_at_1', hue='dataset', data=agg_df, palette="mako")
    
    plt.title('Average Pass@1 by Model and Dataset\n(Averaged across Plan Types)', fontsize=17)
    plt.xlabel('Model Name', fontsize=15)
    plt.ylabel('Average Pass@1 Score (%)', fontsize=15)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    plt.ylim(0, 100)
    plt.legend(title='Dataset', fontsize=11, title_fontsize=13)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "performance_by_model_dataset.png")
    plt.savefig(plot_path)
    print(f"Saved plot: {plot_path}")
    plt.close()

def plot_performance_by_model_dataset_plan(df: pd.DataFrame, output_dir: str):
    """Plots pass@1 by model, dataset, and plan type using catplot for faceting."""
    if df.empty:
        print("Skipping detailed performance plot: DataFrame is empty.")
        return

    # Ensure model_name is treated as categorical for consistent ordering in catplot
    df['model_name'] = pd.Categorical(df['model_name'], categories=df['model_name'].unique(), ordered=True)

    g = sns.catplot(
        x='model_name', 
        y='pass_at_1', 
        hue='plan_type', 
        col='dataset',
        data=df, 
        kind='bar', 
        palette="rocket",
        height=7, 
        aspect=1.7, # Further increased aspect ratio for more width per subplot
        legend=True # Keep default legend initially, we will move it
    )
    
    g.set_axis_labels("Model Name", "Pass@1 Score (%)", fontsize=14) # Label font size
    g.set_titles("{col_name}", size=16) # Subplot title font size 
    g.set_xticklabels(rotation=45, ha='right', fontsize=11)
    g.set(ylim=(0, 100))
    
    # Adjust main title and its position
    g.fig.suptitle('Pass@1 by Model, Plan Type, and Dataset', fontsize=20, y=1.02) # y=1.02 for a bit more space
    
    # Move and style the existing legend
    if g.legend is not None: # Check if legend exists
        g.legend.set_title('Plan Type')
        # Attempt to move legend to top right corner of the figure
        # sns.move_legend(g, "upper right", bbox_to_anchor=(.98, .98))
        plt.setp(g.legend.get_texts(), fontsize='11') 
        plt.setp(g.legend.get_title(), fontsize='13')
        # Position legend outside the plot area if needed or use bbox_to_anchor
        # For catplot, often better to adjust figure layout to make space
    
    # Use fig.tight_layout() with rect to provide padding
    # rect=[left, bottom, right, top]
    g.fig.tight_layout(rect=[0, 0.05, 0.9, 0.95]) # Adjust right to make space for legend, bottom for x-labels

    # Alternative legend positioning if the above is tricky with FacetGrid:
    # handles, labels = g.axes.flatten()[0].get_legend_handles_labels()
    # if handles and labels:
    #     g.fig.legend(handles, labels, title='Plan Type', loc='center right', bbox_to_anchor=(1.05, 0.5), title_fontsize='13', fontsize='11')
    #     g.fig.subplots_adjust(right=0.85) # Make space for legend outside
    
    plot_path = os.path.join(output_dir, "detailed_performance_by_model_dataset_plan.png")
    plt.savefig(plot_path)
    print(f"Saved plot: {plot_path}")
    plt.close()

def load_nlgeval_scores(file_path):
    """Loads NLGEval scores from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            scores = json.load(f)
        return scores
    except FileNotFoundError:
        print(f"Warning: NLGEval scores file not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from NLGEval scores file: {file_path}")
        return None

def plot_nlgeval_comparison(output_dir: str, base_dir: str):
    """Plots a comparison of NLGEval scores for baseline vs. finetuned TinyLlama, separated by dataset."""
    metrics_to_plot = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr"]
    
    paths = {
        "Baseline HumanEval": os.path.join(base_dir, "ollama_baseline_results/test_humaneval/humaneval_ollama_nlgeval_scores.json"),
        "Finetuned HumanEval": os.path.join(base_dir, "save_model_llama3.2_fast_eval/test_humaneval/humaneval_nlgeval_scores.json"),
        "Baseline OpenEval": os.path.join(base_dir, "ollama_baseline_results/test_openeval/openeval_ollama_nlgeval_scores.json"),
        "Finetuned OpenEval": os.path.join(base_dir, "save_model_llama3.2_fast_eval/test_openeval/openeval_nlgeval_scores.json")
    }

    all_data_for_plot = []
    all_scores_loaded_successfully = True
    for label, path in paths.items():
        scores = load_nlgeval_scores(path)
        if scores:
            dataset_type = "HumanEval" if "HumanEval" in label else "OpenEval"
            model_type = "Baseline" if "Baseline" in label else "Finetuned"
            for metric in metrics_to_plot:
                if metric in scores:
                    all_data_for_plot.append({
                        'Dataset': dataset_type,
                        'Model Type': model_type,
                        'Metric': metric,
                        'Score': scores[metric]
                    })
                else:
                    print(f"Warning: Metric '{metric}' not found in {path}")
        else:
            all_scores_loaded_successfully = False
    
    if not all_data_for_plot or not all_scores_loaded_successfully:
        print("Skipping NLGEval comparison plots: Not enough data or some files were missing/invalid.")
        return

    full_plot_df = pd.DataFrame(all_data_for_plot)

    for dataset in ["HumanEval", "OpenEval"]:
        dataset_df = full_plot_df[full_plot_df['Dataset'] == dataset]
        if dataset_df.empty:
            print(f"No data to plot for NLGEval {dataset} comparison.")
            continue

        plt.figure(figsize=(15, 8)) # Adjusted figure size for better readability
        sns.barplot(x='Metric', y='Score', hue='Model Type', data=dataset_df, palette="coolwarm")
        
        plt.title(f'TinyLlama NLGEval Scores - {dataset} (Baseline vs. Finetuned)', fontsize=16)
        plt.xlabel('Evaluation Metric', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(title='Model Type', title_fontsize='12', fontsize='10')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        plot_filename = f"nlgeval_comparison_{dataset.lower()}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path)
        print(f"Saved plot: {plot_path}")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate plots from pass@1 summary CSV and NLGEval scores.")
    parser.add_argument(
        "--csv_path", 
        type=str, 
        default=DEFAULT_CSV_PATH,
        help=f"Path to the pass_at_1_summary.csv file (default: {DEFAULT_CSV_PATH})"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_PLOTS_DIR,
        help=f"Directory to save plots (default: {DEFAULT_PLOTS_DIR})"
    )
    args = parser.parse_args()

    # Determine the base directory (COTTON/)
    # Assumes plot_results.py is in COTTON/RQ1_Recreation/
    script_dir = os.path.dirname(__file__)
    base_dir = os.path.abspath(os.path.join(script_dir, "..")) 

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    # Plot pass@1 scores
    if os.path.exists(args.csv_path):
        try:
            df = pd.read_csv(args.csv_path)
            
            # Filter out Starcoder models before any plotting
            if 'model_name' in df.columns:
                original_models = df['model_name'].nunique()
                df = df[~df['model_name'].str.contains("starcoder", case=False, na=False)]
                if df.empty and original_models > 0:
                    print("DataFrame is empty after filtering out Starcoder models. No pass@1 plots will be generated.")
                    # Still allow NLGEval plots to proceed if only Starcoder data was present for pass@1
                elif df.empty and original_models == 0:
                    print("DataFrame is empty and contained no models to begin with. No pass@1 plots.")
                    # Pass an empty df to plotting functions so they print their skip messages
                # If df is not empty after filtering, proceed to plot pass@1
            else:
                print("Warning: 'model_name' column not found in CSV. Cannot filter Starcoder models for pass@1 plots.")

            if not df.empty and 'pass_at_1' in df.columns:
                df['pass_at_1'] = pd.to_numeric(df['pass_at_1'], errors='coerce')
                df.dropna(subset=['pass_at_1'], inplace=True) # Drop rows where conversion failed
                if not df.empty: # Check again after dropna
                    plot_overall_performance_by_model(df.copy(), args.output_dir)
                    plot_performance_by_model_and_dataset(df.copy(), args.output_dir)
                    plot_performance_by_model_dataset_plan(df.copy(), args.output_dir)
                else:
                    print("Pass@1 DataFrame is empty after to_numeric conversion or dropna. Skipping pass@1 plots.")
            elif not df.empty:
                 print(f"Warning: 'pass_at_1' column not found in {args.csv_path} or DataFrame became empty after filtering. Skipping pass@1 plots.")

        except Exception as e:
            print(f"Error processing CSV file {args.csv_path}: {e}. Skipping pass@1 plots.")
    else:
        print(f"Warning: CSV file not found at {args.csv_path}. Skipping pass@1 plots.")

    # Plot NLGEval comparison
    plot_nlgeval_comparison(args.output_dir, base_dir)

    print("\nPlot generation process finished.")

if __name__ == "__main__":
    main() 