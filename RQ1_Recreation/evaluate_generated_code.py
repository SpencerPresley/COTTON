from typing import Dict
import json
import os
import argparse
import pandas as pd
import re
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, Counter

# Attempt to import human_eval specific functions
try:
    from human_eval.data import read_problems, stream_jsonl
    from human_eval.execution import check_correctness
    HUMAN_EVAL_AVAILABLE = True
except ImportError:
    HUMAN_EVAL_AVAILABLE = False
    print("Warning: human_eval library not found. Please install it to run evaluations.")
    print("Try: pip install human-eval")

def extract_python_code(raw_completion: str) -> str:
    """Extracts Python code from a raw LLM completion, often in markdown blocks."""
    if not isinstance(raw_completion, str):
        return ""

    # Prioritize standard markdown code blocks
    match_python = re.search(r"```python\s*\n(.*?)\n```", raw_completion, re.DOTALL)
    if match_python:
        return match_python.group(1).strip()
    
    match_generic = re.search(r"```\s*\n(.*?)\n```", raw_completion, re.DOTALL)
    if match_generic:
        return match_generic.group(1).strip()

    # If no markdown blocks, check if the whole thing looks like code
    stripped_completion = raw_completion.strip()
    lines = stripped_completion.splitlines()

    if not lines:
        return ""

    first_significant_line_index = -1
    for i, line in enumerate(lines):
        if line.strip():
            first_significant_line_index = i
            break
    
    if first_significant_line_index != -1 and \
       lines[first_significant_line_index].strip().startswith(("def ", "class ", "import ", "from ")):
        return stripped_completion
    
    if len(lines) == 1 and stripped_completion and "```" not in raw_completion:
        return stripped_completion

    print(f"Warning: Could not reliably extract Python code. Input (first 200 chars): {repr(raw_completion[:200])}...")
    return stripped_completion


def evaluate_single_file(
    generation_file_path: str, 
    benchmark_problems: Dict[str, Dict],
    timeout: float,
    num_workers: int
) -> pd.DataFrame:
    
    detailed_results = []
    samples_to_evaluate = []
    
    print(f"Reading and cleaning samples from: {generation_file_path}")
    for i, sample in enumerate(stream_jsonl(generation_file_path)):
        task_id = sample.get('task_id')
        # Use raw_completion as the primary source for extraction, 
        # as generated_code might have been previously (mis)processed by the generation script's extractor
        raw_code_from_generation = sample.get('raw_completion') 
        if not raw_code_from_generation: # Fallback if raw_completion is empty but generated_code field exists
             raw_code_from_generation = sample.get('generated_code')

        if not task_id or not raw_code_from_generation:
            print(f"Skipping sample {i+1} in {generation_file_path} due to missing task_id or code content.")
            continue

        if raw_code_from_generation == "ERROR: Generation failed":
            print(f"Skipping {task_id} from {generation_file_path} due to prior generation error.")
            detailed_results.append({
                'task_id': task_id,
                'passed': False,
                'result': sample.get('error', 'ERROR: Generation failed during previous step'),
                'cleaned_code': raw_code_from_generation, 
                'original_file': os.path.basename(generation_file_path)
            })
            continue
            
        cleaned_code = extract_python_code(raw_code_from_generation)
        problem_details = benchmark_problems.get(task_id)

        if not problem_details:
            print(f"Warning: No benchmark problem found for task_id '{task_id}' from {generation_file_path}. Skipping.")
            continue
        
        samples_to_evaluate.append({
            'task_id': task_id,
            'problem_details': problem_details,
            'cleaned_code': cleaned_code,
            'original_file': os.path.basename(generation_file_path)
        })

    if not samples_to_evaluate:
        print(f"No valid samples found to evaluate in {generation_file_path}")
        return pd.DataFrame(detailed_results)

    print(f"Executing {len(samples_to_evaluate)} samples from {generation_file_path}...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_sample_info = {} 
        for sample_eval_info in samples_to_evaluate:
            args = (
                sample_eval_info['problem_details'], 
                sample_eval_info['cleaned_code'], 
                timeout, 
            )
            future = executor.submit(check_correctness, *args)
            future_to_sample_info[future] = sample_eval_info 

        for future in as_completed(future_to_sample_info): 
            sample_info_back = future_to_sample_info[future] 
            try:
                eval_result = future.result()
                detailed_results.append({
                    'task_id': sample_info_back['task_id'],
                    'passed': eval_result['passed'],
                    'result': eval_result['result'],
                    'cleaned_code': sample_info_back['cleaned_code'],
                    'original_file': sample_info_back['original_file']
                })
            except Exception as e:
                print(f"Error during execution for task {sample_info_back['task_id']}: {e}")
                detailed_results.append({
                    'task_id': sample_info_back['task_id'],
                    'passed': False,
                    'result': f'execution_error: {e}',
                    'cleaned_code': sample_info_back['cleaned_code'],
                    'original_file': sample_info_back['original_file']
                })
                
    return pd.DataFrame(detailed_results)

def main():
    if not HUMAN_EVAL_AVAILABLE:
        print("Exiting: human_eval library is required for evaluation.")
        return

    parser = argparse.ArgumentParser(description="Evaluate generated code for HumanEval/OpenEval benchmarks.")
    parser.add_argument("--results_base_dir", type=str, default="results",
                        help="Base directory where model generation results are stored.")
    parser.add_argument("--benchmark_data_dir", type=str, default=".",
                        help="Directory containing HumanEval.jsonl and OpenEval.jsonl.")
    parser.add_argument("--output_summary_csv", type=str, default="pass_at_1_summary.csv",
                        help="Path to save the aggregated pass@1 scores.")
    parser.add_argument("--output_detailed_results_csv", type=str, default="detailed_evaluation_results.csv",
                        help="Path to save detailed evaluation results for each problem.")
    parser.add_argument("--timeout", type=float, default=5.0, help="Timeout for code execution in seconds.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for parallel execution.")

    args = parser.parse_args()

    # Ensure the script base directory is correctly determined for relative paths
    script_base_dir = os.path.dirname(os.path.realpath(__file__))
    results_base_dir_abs = os.path.join(script_base_dir, args.results_base_dir)
    benchmark_data_dir_abs = os.path.join(script_base_dir, "..", "RQ3") # Assuming RQ3 is one level up from RQ1_Recreation
    output_summary_csv_abs = os.path.join(script_base_dir, args.output_summary_csv)
    output_detailed_results_csv_abs = os.path.join(script_base_dir, args.output_detailed_results_csv)

    # Find all generation files
    generation_files = glob.glob(os.path.join(results_base_dir_abs, "*", "*", "*_plan_generations.jsonl"), recursive=True)

    if not generation_files:
        print(f"No generation files found in {results_base_dir_abs}. Exiting.")
        return
    
    print(f"Found {len(generation_files)} generation files to evaluate.")

    # Load benchmark problems once
    try:
        humaneval_problems = read_problems(os.path.join(benchmark_data_dir_abs, "HumanEval.jsonl"))
        openeval_problems = read_problems(os.path.join(benchmark_data_dir_abs, "OpenEval.jsonl"))
        print(f"Loaded {len(humaneval_problems)} HumanEval problems and {len(openeval_problems)} OpenEval problems.")
    except FileNotFoundError as e:
        print(f"Error loading benchmark problem files: {e}. Please check paths in --benchmark_data_dir (resolved to: {benchmark_data_dir_abs}).")
        return
    except Exception as e:
        print(f"An unexpected error occurred loading benchmark files: {e}")
        return

    all_detailed_results = []
    pass_at_1_summary = []

    for gen_file in generation_files:
        print(f"\nProcessing file: {gen_file}")
        path_parts = gen_file.replace(results_base_dir_abs, '').strip(os.sep).split(os.sep)
        try:
            model_name = path_parts[0]
            dataset_type = path_parts[1]
            plan_type_full = path_parts[2]
            plan_type = plan_type_full.split('_plan_generations.jsonl')[0]
        except IndexError:
            print(f"Could not parse model/dataset/plan from path: {gen_file}. Skipping.")
            continue
        
        current_benchmark_problems = None
        if dataset_type == "humaneval":
            current_benchmark_problems = humaneval_problems
        elif dataset_type == "openeval":
            current_benchmark_problems = openeval_problems
        else:
            print(f"Unknown dataset type '{dataset_type}' inferred from path {gen_file}. Skipping.")
            continue

        if not current_benchmark_problems:
            print(f"Benchmark problems not loaded for {dataset_type}. Skipping file {gen_file}.")
            continue

        file_results_df = evaluate_single_file(gen_file, current_benchmark_problems, args.timeout, args.num_workers)
        
        if file_results_df is not None and not file_results_df.empty:
            file_results_df['model_name'] = model_name
            file_results_df['dataset'] = dataset_type
            file_results_df['plan_type'] = plan_type
            all_detailed_results.append(file_results_df)

            total_problems = len(file_results_df)
            passed_problems = file_results_df['passed'].sum()
            pass_at_1 = (passed_problems / total_problems) * 100 if total_problems > 0 else 0
            
            print(f"Results for {model_name} on {dataset_type} with '{plan_type}' plan: Pass@1 = {pass_at_1:.2f}% ({passed_problems}/{total_problems})")
            pass_at_1_summary.append({
                'model_name': model_name,
                'dataset': dataset_type,
                'plan_type': plan_type,
                'pass_at_1': pass_at_1,
                'passed_count': passed_problems,
                'total_count': total_problems
            })
        else:
            print(f"No results generated or processed for {gen_file}")
            pass_at_1_summary.append({
                'model_name': model_name,
                'dataset': dataset_type,
                'plan_type': plan_type,
                'pass_at_1': 0,
                'passed_count': 0,
                'total_count': 0 # Or should be len(current_benchmark_problems) if we know it tried?
            })

    if all_detailed_results:
        final_detailed_df = pd.concat(all_detailed_results, ignore_index=True)
        final_detailed_df.to_csv(output_detailed_results_csv_abs, index=False)
        print(f"\nDetailed evaluation results saved to {output_detailed_results_csv_abs}")
    else:
        print("\nNo detailed results were generated across all files.")

    if pass_at_1_summary:
        summary_df = pd.DataFrame(pass_at_1_summary)
        summary_df.to_csv(output_summary_csv_abs, index=False)
        print(f"Pass@1 summary saved to {output_summary_csv_abs}")
        print("\nSummary of Pass@1 Scores:")
        print(summary_df)
    else:
        print("\nNo summary scores were generated.")

if __name__ == "__main__":
    main() 