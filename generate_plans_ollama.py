import pandas as pd
from tqdm import tqdm
import argparse
import os

# Attempt to import Langchain components
try:
    from langchain_ollama.llms import OllamaLLM
    from langchain_core.prompts import PromptTemplate
except ImportError:
    print("Langchain components not found. Please ensure langchain, langchain-community are installed.")
    print("Try: pip install langchain langchain-community")
    exit(1)

# Define default paths relative to the COTTON directory or a common base
DEFAULT_INPUT_DIR = "dataset"
DEFAULT_OLLAMA_OUTPUT_BASE_DIR = "ollama_baseline_results"

DEFAULT_HUMANEVAL_INPUT_CSV = os.path.join(DEFAULT_INPUT_DIR, "humaneval.csv")
DEFAULT_HUMANEVAL_OUTPUT_DIR = os.path.join(DEFAULT_OLLAMA_OUTPUT_BASE_DIR, "test_humaneval")

DEFAULT_OPENEVAL_INPUT_CSV = os.path.join(DEFAULT_INPUT_DIR, "openeval.csv")
DEFAULT_OPENEVAL_OUTPUT_DIR = os.path.join(DEFAULT_OLLAMA_OUTPUT_BASE_DIR, "test_openeval")

# Define the prompt template including a few-shot example
# This helps guide the base model to produce the desired "How to solve" output format.
FEW_SHOT_EXAMPLE_INPUT = """from typing import List

def below_zero(operations: List[int]) -> bool:
    \"\"\" You're given a list of deposit and withdrawal operations on a bank account that starts with
    zero balance. Your task is to detect if at any point the balance of account falls below zero, and
    at that point function should return True. Otherwise it should return False.
    \"\"\"
    """

FEW_SHOT_EXAMPLE_OUTPUT = """How to solve:
Step 1. Initialize account balance as 0.
Step 2. Iterate through operations.
    -add value to account balance.
    -If account balance < 0, return True.
Step 3. Return False."""

PROMPT_DEFINITION = f"""

### Given a piece of code, output the corresponding implementation idea.

### Example:

#### Input:
{FEW_SHOT_EXAMPLE_INPUT}

#### Output:
{FEW_SHOT_EXAMPLE_OUTPUT}

### Input:
{{src_content}}

#### Output:"""

def generate_plan_with_langchain_ollama(llm, prompt_template, src_content_value):
    """Generates a plan using the initialized LLM and prompt template."""
    try:
        # Create the full prompt by formatting the template with the specific source content
        formatted_prompt = prompt_template.format(src_content=src_content_value)
        
        # Invoke the LLM
        response = llm.invoke(formatted_prompt)
        return response.strip()
    except Exception as e:
        print(f"Error during Langchain Ollama invocation for prompt \"{src_content_value[:50]}...\": {e}")
        return "Error: Could not generate prediction via Langchain Ollama."

def main():
    parser = argparse.ArgumentParser(description="Generate plans using a base LLM via Langchain and Ollama.")
    
    path_group = parser.add_argument_group('Manual Path Specification')
    path_group.add_argument("--input_csv", type=str, help="Path to the input CSV file (e.g., dataset/humaneval.csv)")
    path_group.add_argument("--output_dir", type=str, help="Directory to save the predictions.csv file.")
    
    parser.add_argument("--ollama_model", type=str, default="tinyllama:1.1b", help="Name of the model in Ollama (e.g., tinyllama, mistral).")
    parser.add_argument("--src_column", type=str, default="src", help="Name of the column in input_csv containing the source code prompts.")
    parser.add_argument("--ollama_base_url", type=str, default="http://localhost:11434", help="Base URL for the Ollama API.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for Ollama model generation.")

    default_run_group = parser.add_mutually_exclusive_group()
    default_run_group.add_argument("--run_humaneval_default", action="store_true", help=f"Run with default HumanEval input ({DEFAULT_HUMANEVAL_INPUT_CSV}) and output to {DEFAULT_HUMANEVAL_OUTPUT_DIR}")
    default_run_group.add_argument("--run_openeval_default", action="store_true", help=f"Run with default OpenEval input ({DEFAULT_OPENEVAL_INPUT_CSV}) and output to {DEFAULT_OPENEVAL_OUTPUT_DIR}")

    args = parser.parse_args()

    input_csv_to_use = args.input_csv
    output_dir_to_use = args.output_dir

    if args.run_humaneval_default:
        print("Using default paths for HumanEval baseline generation...")
        input_csv_to_use = DEFAULT_HUMANEVAL_INPUT_CSV
        # If output_dir is also specified with the default flag, let it override the default output for this run type
        output_dir_to_use = args.output_dir if args.output_dir else DEFAULT_HUMANEVAL_OUTPUT_DIR

    elif args.run_openeval_default:
        print("Using default paths for OpenEval baseline generation...")
        input_csv_to_use = DEFAULT_OPENEVAL_INPUT_CSV
        output_dir_to_use = args.output_dir if args.output_dir else DEFAULT_OPENEVAL_OUTPUT_DIR

    if not input_csv_to_use or not output_dir_to_use:
        if not (args.run_humaneval_default or args.run_openeval_default):
            parser.error("Either specify --input_csv and --output_dir, or use one of the --run_..._default flags.")
        # This case should ideally be caught if default paths are None, but good to have a fallback.
        elif not input_csv_to_use:
            print("Error: Input CSV path is missing for the selected default run.")
            return
        elif not output_dir_to_use:
            print("Error: Output directory path is missing for the selected default run.")
            return

    if not os.path.exists(input_csv_to_use):
        print(f"Error: Input CSV not found at {input_csv_to_use}")
        return

    # output_dir_to_use will be created if it doesn't exist
    if not os.path.exists(output_dir_to_use):
        os.makedirs(output_dir_to_use)
        print(f"Created output directory: {output_dir_to_use}")

    df_input = pd.read_csv(input_csv_to_use)
    if args.src_column not in df_input.columns:
        print(f"Error: Source column '{args.src_column}' not found in {input_csv_to_use}")
        return
    
    source_contents = df_input[args.src_column].tolist()
    predictions = []

    try:
        print(f"Initializing Ollama model: {args.ollama_model} via Langchain...")
        llm = OllamaLLM(
            base_url=args.ollama_base_url, 
            model=args.ollama_model,
            temperature=args.temperature
            # Add other parameters like top_p if needed, e.g., top_p=0.95
        )
        # Test connection with a simple invoke
        print("Testing Ollama connection with a short prompt...")
        llm.invoke("Say hi.") 
        print("Ollama connection successful.")

    except Exception as e:
        print(f"Error initializing Ollama model or connecting: {e}")
        print("Please ensure Ollama is running and the model is available.")
        return

    # Define the prompt template using Langchain
    prompt_template = PromptTemplate(
        input_variables=["src_content"],
        template=PROMPT_DEFINITION
    )

    print(f"Generating plans for {len(source_contents)} prompts using Langchain Ollama model: {args.ollama_model}...")
    for src_text in tqdm(source_contents):
        prediction = generate_plan_with_langchain_ollama(llm, prompt_template, src_text)
        predictions.append(prediction)
    
    df_predictions = pd.DataFrame(predictions)
    output_filename = "predictions.csv"
    output_predictions_csv = os.path.join(output_dir_to_use, output_filename)
    df_predictions.to_csv(output_predictions_csv, index=False, header=None)
    print(f"Saved {len(predictions)} predictions to {output_predictions_csv}")

if __name__ == "__main__":
    main()
