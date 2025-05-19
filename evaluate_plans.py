import pandas as pd
from nlgeval import NLGEval
import argparse
import os
import json

# Define default paths relative to the COTTON directory or a common base
DEFAULT_FINETUNED_OUTPUT_DIR = "save_model_llama3.2_fast_eval" # For gold standards from fine-tuned run
DEFAULT_OLLAMA_OUTPUT_BASE_DIR = "ollama_baseline_results" # For predictions from ollama run

# Gold files are from the fine-tuned model's test run (as they are the reference texts)
DEFAULT_HUMANEVAL_GOLD = os.path.join(DEFAULT_FINETUNED_OUTPUT_DIR, "test_humaneval/gold.csv")
DEFAULT_OPENEVAL_GOLD = os.path.join(DEFAULT_FINETUNED_OUTPUT_DIR, "test_openeval/gold.csv")

# Defaults for fine-tuned model's predictions & scores (original defaults)
DEFAULT_HUMANEVAL_FINETUNED_PREDS = os.path.join(DEFAULT_FINETUNED_OUTPUT_DIR, "test_humaneval/predictions.csv")
DEFAULT_HUMANEVAL_FINETUNED_OUTPUT_JSON = os.path.join(DEFAULT_FINETUNED_OUTPUT_DIR, "test_humaneval/humaneval_nlgeval_scores.json")

DEFAULT_OPENEVAL_FINETUNED_PREDS = os.path.join(DEFAULT_FINETUNED_OUTPUT_DIR, "test_openeval/predictions.csv")
DEFAULT_OPENEVAL_FINETUNED_OUTPUT_JSON = os.path.join(DEFAULT_FINETUNED_OUTPUT_DIR, "test_openeval/openeval_nlgeval_scores.json")

# Defaults for Ollama baseline predictions & scores
DEFAULT_HUMANEVAL_OLLAMA_PREDS = os.path.join(DEFAULT_OLLAMA_OUTPUT_BASE_DIR, "test_humaneval/predictions.csv")
DEFAULT_HUMANEVAL_OLLAMA_OUTPUT_JSON = os.path.join(DEFAULT_OLLAMA_OUTPUT_BASE_DIR, "test_humaneval/humaneval_ollama_nlgeval_scores.json")

DEFAULT_OPENEVAL_OLLAMA_PREDS = os.path.join(DEFAULT_OLLAMA_OUTPUT_BASE_DIR, "test_openeval/predictions.csv")
DEFAULT_OPENEVAL_OLLAMA_OUTPUT_JSON = os.path.join(DEFAULT_OLLAMA_OUTPUT_BASE_DIR, "test_openeval/openeval_ollama_nlgeval_scores.json")

def main():
    parser = argparse.ArgumentParser(description="Evaluate generated plans using nlgeval.")
    
    path_group = parser.add_argument_group('Manual Path Specification')
    path_group.add_argument("--gold_file", type=str, help="Path to CSV file with gold standard plans.")
    path_group.add_argument("--predictions_file", type=str, help="Path to CSV file with predicted plans.")
    path_group.add_argument("--output_json_file", type=str, help="Optional path to save metrics as JSON.")

    default_run_group = parser.add_mutually_exclusive_group()
    default_run_group.add_argument("--run_humaneval_finetuned_default", action="store_true", help=f"Eval fine-tuned HumanEval: gold from {DEFAULT_HUMANEVAL_GOLD}, preds from {DEFAULT_HUMANEVAL_FINETUNED_PREDS}")
    default_run_group.add_argument("--run_openeval_finetuned_default", action="store_true", help=f"Eval fine-tuned OpenEval: gold from {DEFAULT_OPENEVAL_GOLD}, preds from {DEFAULT_OPENEVAL_FINETUNED_PREDS}")
    default_run_group.add_argument("--run_humaneval_ollama_default", action="store_true", help=f"Eval Ollama HumanEval: gold from {DEFAULT_HUMANEVAL_GOLD}, preds from {DEFAULT_HUMANEVAL_OLLAMA_PREDS}")
    default_run_group.add_argument("--run_openeval_ollama_default", action="store_true", help=f"Eval Ollama OpenEval: gold from {DEFAULT_OPENEVAL_GOLD}, preds from {DEFAULT_OPENEVAL_OLLAMA_PREDS}")
    
    args = parser.parse_args()

    gold_file_to_use = args.gold_file
    predictions_file_to_use = args.predictions_file
    output_json_to_use = args.output_json_file
    run_description = "custom paths"

    if args.run_humaneval_finetuned_default:
        run_description = "HumanEval (Fine-tuned Model)"
        print(f"Using default paths for {run_description} evaluation...")
        gold_file_to_use = DEFAULT_HUMANEVAL_GOLD
        predictions_file_to_use = DEFAULT_HUMANEVAL_FINETUNED_PREDS
        output_json_to_use = args.output_json_file if args.output_json_file else DEFAULT_HUMANEVAL_FINETUNED_OUTPUT_JSON

    elif args.run_openeval_finetuned_default:
        run_description = "OpenEval (Fine-tuned Model)"
        print(f"Using default paths for {run_description} evaluation...")
        gold_file_to_use = DEFAULT_OPENEVAL_GOLD
        predictions_file_to_use = DEFAULT_OPENEVAL_FINETUNED_PREDS
        output_json_to_use = args.output_json_file if args.output_json_file else DEFAULT_OPENEVAL_FINETUNED_OUTPUT_JSON

    elif args.run_humaneval_ollama_default:
        run_description = "HumanEval (Ollama Baseline)"
        print(f"Using default paths for {run_description} evaluation...")
        gold_file_to_use = DEFAULT_HUMANEVAL_GOLD # Gold is still the same reference
        predictions_file_to_use = DEFAULT_HUMANEVAL_OLLAMA_PREDS
        output_json_to_use = args.output_json_file if args.output_json_file else DEFAULT_HUMANEVAL_OLLAMA_OUTPUT_JSON

    elif args.run_openeval_ollama_default:
        run_description = "OpenEval (Ollama Baseline)"
        print(f"Using default paths for {run_description} evaluation...")
        gold_file_to_use = DEFAULT_OPENEVAL_GOLD # Gold is still the same reference
        predictions_file_to_use = DEFAULT_OPENEVAL_OLLAMA_PREDS
        output_json_to_use = args.output_json_file if args.output_json_file else DEFAULT_OPENEVAL_OLLAMA_OUTPUT_JSON

    if not gold_file_to_use or not predictions_file_to_use:
        if not any([args.run_humaneval_finetuned_default, args.run_openeval_finetuned_default, 
                    args.run_humaneval_ollama_default, args.run_openeval_ollama_default]):
            parser.error("Either specify --gold_file and --predictions_file, or use one of the --run_..._default flags.")
        elif not gold_file_to_use:
             print(f"Error: Gold file path is missing for the selected default run ({run_description}).")
             return
        elif not predictions_file_to_use:
             print(f"Error: Predictions file path is missing for the selected default run ({run_description}).")
             return   

    print(f"\n--- Evaluating: {run_description} ---")
    if not os.path.exists(gold_file_to_use):
        print(f"Error: Gold file not found at {gold_file_to_use}")
        return

    if not os.path.exists(predictions_file_to_use):
        print(f"Error: Predictions file not found at {predictions_file_to_use}")
        return

    print(f"Loading gold standards from: {gold_file_to_use}")
    gold_df = pd.read_csv(gold_file_to_use, header=None, names=['text'], keep_default_na=False, na_filter=False)
    references_list = gold_df['text'].astype(str).tolist()
    references_for_nlgeval = [[r] for r in references_list]

    print(f"Loading predictions from: {predictions_file_to_use}")
    pred_df = pd.read_csv(predictions_file_to_use, header=None, names=['text'], keep_default_na=False, na_filter=False)
    hypotheses_list = pred_df['text'].astype(str).tolist()

    if not hypotheses_list:
        print("Error: No hypotheses found in the predictions file.")
        return
    
    if not references_list:
        print("Error: No references found in the gold file.")
        return

    if len(hypotheses_list) != len(references_list):
        print(f"Warning: Number of predictions ({len(hypotheses_list)}) does not match number of gold standards ({len(references_list)}).")
        min_len = min(len(hypotheses_list), len(references_list))
        print(f"Truncating both lists to the minimum length: {min_len}")
        hypotheses_list = hypotheses_list[:min_len]
        references_for_nlgeval = references_for_nlgeval[:min_len] 
        if not hypotheses_list or not references_for_nlgeval:
            print("Error: One of the lists became empty after truncation. Cannot proceed.")
            return

    print("\n--- Debug Info Before NLGEval Call ---")
    print(f"Length of hypotheses_list: {len(hypotheses_list)}")
    print(f"Length of references_for_nlgeval: {len(references_for_nlgeval)}")
    if hypotheses_list:
        print(f"First hypothesis: {hypotheses_list[0][:100]}...") 
    if references_for_nlgeval and references_for_nlgeval[0]:
        print(f"First reference item (should be a list with one string): {references_for_nlgeval[0][0][:100]}...")
    print("-------------------------------------\n")

    hypotheses_list_final = [str(h) for h in hypotheses_list]
    references_for_nlgeval_final = [[str(r_item) for r_item in r_list] for r_list in references_for_nlgeval]

    print("\n--- Debug Info (Final Check) Before NLGEval Call ---")
    print(f"Length of hypotheses_list_final: {len(hypotheses_list_final)}")
    print(f"Type of hypotheses_list_final: {type(hypotheses_list_final)}")
    if hypotheses_list_final:
        print(f"Type of first item in hypotheses_list_final: {type(hypotheses_list_final[0])}")
    print(f"Length of references_for_nlgeval_final: {len(references_for_nlgeval_final)}")
    print(f"Type of references_for_nlgeval_final: {type(references_for_nlgeval_final)}")
    if references_for_nlgeval_final:
        print(f"Type of first item in references_for_nlgeval_final: {type(references_for_nlgeval_final[0])}")
        if references_for_nlgeval_final[0]:
            print(f"Type of first sub-item in references_for_nlgeval_final: {type(references_for_nlgeval_final[0][0])}")
    print("-----------------------------------------------------\n")

    print("\n--- Debugging Empty Strings ---")
    print(f"Original hyps length: {len(hypotheses_list_final)}")
    print(f"Original refs length: {len(references_for_nlgeval_final)}")

    temp_hyps = []
    temp_refs = []

    for i in range(len(hypotheses_list_final)):
        h = hypotheses_list_final[i]
        r_list = references_for_nlgeval_final[i]
        if h.strip() and r_list and r_list[0].strip(): 
            temp_hyps.append(h)
            temp_refs.append(r_list)
        else:
            print(f"Found empty/whitespace string at index {i}. Hyp: '{h[:50]}...', Ref: '{r_list[0][:50]}...'")

    print(f"Length after filtering empty/whitespace: Hyps={len(temp_hyps)}, Refs={len(temp_refs)}")
    print("---------------------------------")

    if not temp_hyps or not temp_refs:
        print("Error: One or both lists became empty after filtering for whitespace. Cannot proceed.")
        return
    
    if len(temp_hyps) != len(temp_refs):
        print(f"Error: Lengths still mismatch after filtering for whitespace. Hyps: {len(temp_hyps)}, Refs: {len(temp_refs)}")
        return

    hypotheses_to_use = temp_hyps
    references_to_use = temp_refs

    print(f"Using {len(hypotheses_to_use)} hypothesis-reference pairs for NLGEval.")

    print("Initializing NLGEval...")
    nlgeval = NLGEval(no_skipthoughts=True, no_glove=True, metrics_to_omit=['SPICE']) # METEOR can be included

    print("Computing metrics...")
    metrics_dict = nlgeval.compute_metrics(ref_list=references_to_use, hyp_list=hypotheses_to_use)
    
    print("\nEvaluation Metrics:")
    for metric, score in metrics_dict.items():
        print(f"  {metric}: {score}")

    if output_json_to_use:
        output_json_dir = os.path.dirname(output_json_to_use)
        if output_json_dir and not os.path.exists(output_json_dir):
            os.makedirs(output_json_dir)
            print(f"Created directory: {output_json_dir}")
            
        print(f"\nSaving metrics to {output_json_to_use}...")
        try:
            with open(output_json_to_use, 'w') as f:
                json.dump(metrics_dict, f, indent=4)
            print(f"Metrics successfully saved to {output_json_to_use}")
        except IOError as e:
            print(f"Error: Could not write metrics to {output_json_to_use}: {e}")

if __name__ == '__main__':
    main() 