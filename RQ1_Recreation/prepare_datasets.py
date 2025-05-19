import pandas as pd
import json
import argparse
import os
import re # For parsing function names
from human_eval.data import read_problems

# Helper function to extract function name from a prompt string
def extract_function_name(prompt_str):
    match = re.search(r"def\s+(\w+)\s*\(", prompt_str)
    if match:
        return match.group(1)
    # print(f"Debug: Could not extract function name from: {prompt_str[:100]}...") # Keep this for now
    return None

def get_docstring_first_line(prompt_str):
    # Adjusted regex to be less greedy and handle various docstring formats
    match = re.search(r'"""\s*(.*?)(?:\n|""")', prompt_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def prepare_dataset(problem_file_path, baseline_predictions_path, 
                    finetuned_predictions_path, original_benchmark_file_path, 
                    output_file_path, dataset_name_prefix):
    """
    Prepares a consolidated JSONL dataset for code generation experiments.
    Uses the original benchmark JSONL file as the source of truth for task_id, prompt, and reference_code.
    Aligns with problem_file_path (for reference_plan) and prediction files (for generated plans).
    """
    try:
        problems_csv_df = pd.read_csv(problem_file_path)
        baseline_plans_df = pd.read_csv(baseline_predictions_path, header=None, names=['baseline_plan'])
        finetuned_plans_df = pd.read_csv(finetuned_predictions_path, header=None, names=['finetuned_plan'])

        original_benchmark_problems = read_problems(original_benchmark_file_path)
        print(f"Loaded {len(original_benchmark_problems)} problems from {os.path.basename(original_benchmark_file_path)}.")

        # Build a dictionary of CSV problems, keyed by their entry_point for efficient lookup
        csv_problems_by_entry_point = {}
        for i, row in problems_csv_df.iterrows():
            src_prompt = row['src']
            entry_point = extract_function_name(src_prompt)
            if entry_point:
                if entry_point not in csv_problems_by_entry_point:
                    csv_problems_by_entry_point[entry_point] = []
                csv_problems_by_entry_point[entry_point].append({
                    'original_index': i,
                    'src_prompt_raw': src_prompt,
                    'src_prompt_stripped': src_prompt.strip(),
                    'reference_plan': row['tgt'],
                    'docstring_fl': get_docstring_first_line(src_prompt) 
                })
            else:
                # This warning is important to see if CSV prompts are failing extraction
                print(f"Warning: Failed to extract entry point from CSV row {i}. Prompt (start): {repr(src_prompt[:150])}...")

        with open(output_file_path, 'w') as outfile:
            aligned_count = 0
            unaligned_benchmark_task_ids = []
            matched_csv_indices = set()

            for task_id, benchmark_entry in original_benchmark_problems.items():
                benchmark_prompt_raw = benchmark_entry['prompt']
                benchmark_prompt_stripped = benchmark_prompt_raw.strip()
                benchmark_entry_point = benchmark_entry.get('entry_point')
                benchmark_docstring_fl = get_docstring_first_line(benchmark_prompt_raw)

                found_match = False
                
                if benchmark_entry_point and benchmark_entry_point in csv_problems_by_entry_point:
                    possible_csv_matches = csv_problems_by_entry_point[benchmark_entry_point]
                    for csv_problem_details in possible_csv_matches:
                        original_csv_index = csv_problem_details['original_index']
                        if original_csv_index in matched_csv_indices:
                            continue 

                        csv_prompt_stripped = csv_problem_details['src_prompt_stripped']
                        csv_docstring_fl = csv_problem_details['docstring_fl']
                        
                        match_criteria_met = False
                        if dataset_name_prefix == 'humaneval':
                            if csv_prompt_stripped in benchmark_prompt_raw: # Check against raw benchmark prompt
                                match_criteria_met = True
                            elif benchmark_docstring_fl and csv_docstring_fl and benchmark_docstring_fl.lower() == csv_docstring_fl.lower():
                                match_criteria_met = True
                                print(f"Debug: Matched {task_id} with CSV idx {original_csv_index} via docstring FL equality: '{csv_docstring_fl}'")
                            # Add a more lenient substring check for docstrings if direct equality fails
                            elif benchmark_docstring_fl and csv_docstring_fl and (csv_docstring_fl.lower() in benchmark_docstring_fl.lower() or benchmark_docstring_fl.lower() in csv_docstring_fl.lower()):
                                match_criteria_met = True
                                print(f"Debug: Matched {task_id} with CSV idx {original_csv_index} via docstring FL substring: '{csv_docstring_fl}' vs '{benchmark_docstring_fl}'")

                        elif dataset_name_prefix == 'openeval':
                            if benchmark_prompt_stripped.endswith(csv_prompt_stripped):
                                match_criteria_met = True
                            elif csv_prompt_stripped == benchmark_prompt_stripped:
                                match_criteria_met = True
                                print(f"Debug: Matched {task_id} via direct stripped prompt equality for OpenEval.")
                        
                        if match_criteria_met:
                            if original_csv_index < len(baseline_plans_df) and original_csv_index < len(finetuned_plans_df):
                                record = {
                                    "task_id": task_id,
                                    "problem_prompt": benchmark_entry['prompt'], 
                                    "reference_plan": csv_problem_details['reference_plan'],
                                    "baseline_tinyllama_plan": baseline_plans_df.iloc[original_csv_index]['baseline_plan'],
                                    "finetuned_tinyllama_plan": finetuned_plans_df.iloc[original_csv_index]['finetuned_plan'],
                                    "reference_code": benchmark_entry['canonical_solution']
                                }
                                outfile.write(json.dumps(record) + '\n')
                                aligned_count += 1
                                matched_csv_indices.add(original_csv_index)
                                found_match = True
                                break 
                            else:
                                print(f"Warning: CSV index {original_csv_index} for matched problem {task_id} is out of bounds for plan prediction files.")
                                found_match = True 
                                break 
                
                if not found_match:
                    unaligned_benchmark_task_ids.append(task_id)
                    # Enhanced Debugging for unaligned tasks
                    if (dataset_name_prefix == 'humaneval' and task_id in ['HumanEval/10', 'HumanEval/32', 'HumanEval/38', 'HumanEval/50', 'HumanEval/66', 'HumanEval/114', 'HumanEval/153']) or \
                       (task_id == "Open/10" and dataset_name_prefix == 'openeval'):
                        print(f"\n--- DEBUG: Persistently Unaligned Benchmark Task: {task_id} ---")
                        print(f"Benchmark Entry Point: {benchmark_entry_point}")
                        print(f"Benchmark Docstring FL: {repr(benchmark_docstring_fl)}")
                        print(f"Benchmark Prompt (raw snippet): {repr(benchmark_prompt_raw[:300])}...")
                        if benchmark_entry_point and benchmark_entry_point in csv_problems_by_entry_point:
                             print(f"  Potentially related CSV prompts for EP '{benchmark_entry_point}' (unmatched indices):")
                             for csv_detail_debug in csv_problems_by_entry_point[benchmark_entry_point]:
                                 if csv_detail_debug['original_index'] not in matched_csv_indices: 
                                     print(f"    CSV Opt (idx {csv_detail_debug['original_index']}) EP: {csv_detail_debug['entry_point']}, DocFL: {repr(csv_detail_debug['docstring_fl'])}, Stripped Prompt: {repr(csv_detail_debug['src_prompt_stripped'][:200])}...")
                        else:
                            print(f"  No CSV prompts found with benchmark entry point '{benchmark_entry_point}' in pre-built dict.")
                        
                        # For Open/10, try to find the specific CSV prompt by its expected index
                        if task_id == "Open/10" and dataset_name_prefix == 'openeval':
                            expected_csv_idx_open10 = 10 # As Open/10 is the 11th problem
                            if expected_csv_idx_open10 < len(problems_csv_df):
                                csv_prompt_open10_raw = problems_csv_df.iloc[expected_csv_idx_open10]['src']
                                print(f"--- Specific Raw CSV Prompt for Open/10 (expected CSV idx {expected_csv_idx_open10}):\n{repr(csv_prompt_open10_raw)}---END CSV---")
                            else:
                                print(f"Could not retrieve openeval.csv row for index {expected_csv_idx_open10} for Open/10 debug.")

            print(f"\nSuccessfully wrote {aligned_count} aligned records to {output_file_path}.")
            
            unmatched_csv_count = len(problems_csv_df) - len(matched_csv_indices)
            if unmatched_csv_count > 0:
                print(f"Warning: {unmatched_csv_count} problems from {os.path.basename(problem_file_path)} could not be mapped to any benchmark problem.")
            
            if unaligned_benchmark_task_ids:
                print(f"Warning: {len(unaligned_benchmark_task_ids)} tasks from {os.path.basename(original_benchmark_file_path)} could not be aligned with CSV problems. First few unaligned task IDs: {unaligned_benchmark_task_ids[:20]}...")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e.filename}")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description="Prepare consolidated JSONL datasets for HumanEval or OpenEval.")
    parser.add_argument("--dataset_type", type=str, required=True, choices=['humaneval', 'openeval'], 
                        help="Type of dataset to prepare ('humaneval' or 'openeval').")
    
    script_dir = os.path.dirname(os.path.realpath(__file__))
    cotton_root_dir = os.path.abspath(os.path.join(script_dir, "..")) 

    base_dataset_path = os.path.join(cotton_root_dir, "dataset")
    base_ollama_results_path = os.path.join(cotton_root_dir, "ollama_baseline_results")
    base_finetuned_results_path = os.path.join(cotton_root_dir, "save_model_llama3.2_fast_eval")
    rq3_data_path = os.path.join(cotton_root_dir, "RQ3") 
    output_data_path = os.path.join(cotton_root_dir, "RQ1_Recreation/data")

    args = parser.parse_args()

    os.makedirs(output_data_path, exist_ok=True)

    if args.dataset_type == 'humaneval':
        problem_file = os.path.join(base_dataset_path, "humaneval.csv")
        baseline_preds = os.path.join(base_ollama_results_path, "test_humaneval/predictions.csv")
        finetuned_preds = os.path.join(base_finetuned_results_path, "test_humaneval/predictions.csv")
        original_benchmark_file = os.path.join(rq3_data_path, "HumanEval.jsonl")
        output_file = os.path.join(output_data_path, "humaneval_prepared.jsonl")
        dataset_prefix = "humaneval"
    elif args.dataset_type == 'openeval':
        problem_file = os.path.join(base_dataset_path, "openeval.csv")
        baseline_preds = os.path.join(base_ollama_results_path, "test_openeval/predictions.csv")
        finetuned_preds = os.path.join(base_finetuned_results_path, "test_openeval/predictions.csv")
        original_benchmark_file = os.path.join(rq3_data_path, "OpenEval.jsonl")
        output_file = os.path.join(output_data_path, "openeval_prepared.jsonl")
        dataset_prefix = "openeval"
    else:
        print(f"Error: Invalid dataset_type '{args.dataset_type}'. Must be 'humaneval' or 'openeval'.")
        return

    print(f"Starting preparation for {args.dataset_type} dataset...")
    print(f"  Problem CSV: {problem_file}")
    print(f"  Baseline plans CSV: {baseline_preds}")
    print(f"  Finetuned plans CSV: {finetuned_preds}")
    print(f"  Original benchmark JSONL: {original_benchmark_file}")
    print(f"  Output JSONL: {output_file}")

    prepare_dataset(problem_file, baseline_preds, finetuned_preds, 
                    original_benchmark_file, output_file, dataset_prefix)

if __name__ == "__main__":
    main() 