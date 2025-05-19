import pandas as pd
import json
import argparse
import os
import re
import time

try:
    from langchain_ollama.llms import OllamaLLM
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: Langchain components not found. Please ensure langchain, langchain-ollama, langchain-openai, langchain-core are installed.")
    print("Try: pip install langchain langchain-ollama langchain-openai langchain-core")

# --- Prompt Templates ---
PROMPT_TEMPLATE_WITH_PLAN = """
You are an expert Python programmer.
Based on the following problem description and implementation plan, please write the Python code solution.
Only provide the complete Python code block. Do not include any explanations or surrounding text.

### Problem:
{problem_prompt}

### Implementation Plan:
{plan_steps}

### Python Code:
"""

PROMPT_TEMPLATE_NO_PLAN = """
You are an expert Python programmer.
Based on the following problem description, please write the Python code solution.
Only provide the complete Python code block. Do not include any explanations or surrounding text.

### Problem:
{problem_prompt}

### Python Code:
"""

# --- Helper Functions ---
def stream_jsonl(filename: str):
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found.")
        return []
    with open(filename, "r") as fp:
        for line in fp:
            if any(not x.isspace() for x in line):
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line in {filename}: {line.strip()}")

def write_jsonl(filename: str, data, append: bool = False):
    mode = 'a' if append else 'w'
    with open(filename, mode) as fp:
        for x in data:
            fp.write(json.dumps(x) + "\n")

def extract_python_code(raw_completion: str) -> str:
    """Extracts Python code from a raw LLM completion, often in markdown blocks."""
    # Common markdown code block pattern
    match = re.search(r"```python\n(.*?)```", raw_completion, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # If no markdown, check if the completion itself looks like a plausible start of code
    # (e.g., starts with def, class, import, or typical indentation)
    # This is a heuristic and might need refinement.
    lines = raw_completion.strip().split('\n')
    if lines and (lines[0].startswith("def ") or lines[0].startswith("class ") or lines[0].startswith("import ") or lines[0].startswith("from ")):
        return raw_completion.strip()
        
    # Fallback: if it's just a single block of text that could be code
    # This is less reliable and can be commented out if too aggressive
    # if not "```" in raw_completion and len(raw_completion.splitlines()) > 1:
    #     return raw_completion.strip()

    # If still no match, return the original completion, assuming it might be direct code
    # or to let the user see the raw output if extraction fails.
    print(f"Warning: Could not extract Python code from markdown block. Using raw completion:\n{raw_completion[:200]}...")
    return raw_completion.strip() 

# --- Main Generation Logic ---
def generate_code_for_dataset(
    coder_model_name: str,
    dataset_name: str,
    plan_type: str,
    prepared_data_dir: str,
    output_results_dir: str,
    ollama_base_url: str,
    openai_api_key: str,
    temperature: float,
    max_tokens: int,
    num_samples_per_problem: int = 1 # Default to 1 sample
):
    if not LANGCHAIN_AVAILABLE:
        print("Langchain components are not available. Aborting generation.")
        return

    input_jsonl_path = os.path.join(prepared_data_dir, f"{dataset_name}_prepared.jsonl")
    if not os.path.exists(input_jsonl_path):
        print(f"Error: Prepared dataset not found at {input_jsonl_path}")
        return

    # Initialize LLM
    llm = None
    if coder_model_name.startswith("gpt-"):
        if not openai_api_key:
            print("Error: OpenAI API key not provided for GPT model. Set OPENAI_API_KEY environment variable or use --openai_api_key.")
            return
        try:
            llm = ChatOpenAI(
                model_name=coder_model_name, 
                api_key=openai_api_key, 
                temperature=temperature,
                max_tokens=max_tokens
            )
            print(f"Initialized OpenAI model: {coder_model_name}")
        except Exception as e:
            print(f"Error initializing OpenAI model {coder_model_name}: {e}")
            return
    else: # Ollama model
        try:
            llm = OllamaLLM(
                base_url=ollama_base_url, 
                model=coder_model_name, 
                temperature=temperature
                # OllamaLLM may not directly support max_tokens in constructor, 
                # it's often controlled by model's Modelfile or API call params
                # For Langchain invoke, it might be part of llm.invoke(prompt, stop=["..."], max_tokens=...)
                # or specific config. For now, we assume temperature is the main control for Ollama via Langchain.
            )
            print(f"Initialized Ollama model: {coder_model_name} from {ollama_base_url}")
        except Exception as e:
            print(f"Error initializing Ollama model {coder_model_name}: {e}")
            return

    # Prepare output path
    safe_model_name = coder_model_name.replace(":", "_").replace("/", "_")
    output_dir_for_model = os.path.join(output_results_dir, safe_model_name, dataset_name)
    os.makedirs(output_dir_for_model, exist_ok=True)
    output_jsonl_path = os.path.join(output_dir_for_model, f"{plan_type}_plan_generations.jsonl")

    print(f"Generating code for {dataset_name} using {coder_model_name} with plan type '{plan_type}'.")
    print(f"Output will be saved to: {output_jsonl_path}")

    generated_results = []
    processed_tasks = 0

    for problem_idx, problem_data in enumerate(stream_jsonl(input_jsonl_path)):
        task_id = problem_data['task_id']
        problem_prompt = problem_data['problem_prompt']
        
        plan_to_use = ""
        if plan_type == 'baseline':
            plan_to_use = problem_data.get('baseline_tinyllama_plan', "")
        elif plan_type == 'finetuned':
            plan_to_use = problem_data.get('finetuned_tinyllama_plan', "")

        if plan_type == 'none' or not plan_to_use:
            current_prompt_template = PromptTemplate.from_template(PROMPT_TEMPLATE_NO_PLAN)
            formatted_prompt = current_prompt_template.format(problem_prompt=problem_prompt)
        else:
            current_prompt_template = PromptTemplate.from_template(PROMPT_TEMPLATE_WITH_PLAN)
            formatted_prompt = current_prompt_template.format(problem_prompt=problem_prompt, plan_steps=plan_to_use)
        
        print(f"Processing {task_id} ({problem_idx + 1} / Total in input)...", end=" ")
        
        # For now, num_samples_per_problem = 1. Loop if we want more later.
        # If num_samples_per_problem > 1, ensure LLM temperature is > 0 for varied outputs.
        raw_completion = ""
        for _ in range(num_samples_per_problem):
            try:
                # Note: max_tokens for Ollama might need to be passed differently if supported by the specific Langchain OllamaLLM version
                # For ChatOpenAI, max_tokens is a constructor arg.
                response = llm.invoke(formatted_prompt) 
                raw_completion = response if isinstance(response, str) else response.content
                generated_code = extract_python_code(raw_completion)
                print(f"Success.")

                result_record = {
                    "task_id": task_id,
                    "problem_prompt": problem_prompt,
                    "plan_type_used": plan_type,
                    "plan_steps_used": plan_to_use,
                    "generated_code": generated_code,
                    "raw_completion": raw_completion, # Store raw completion for debugging extraction
                    "reference_code": problem_data.get('reference_code', "")
                }
                generated_results.append(result_record)
                write_jsonl(output_jsonl_path, [result_record], append=(processed_tasks > 0))
                processed_tasks += 1

            except Exception as e:
                print(f"Failed for {task_id}. Error: {e}")
                error_record = {
                    "task_id": task_id,
                    "problem_prompt": problem_prompt,
                    "plan_type_used": plan_type,
                    "plan_steps_used": plan_to_use,
                    "generated_code": "ERROR: Generation failed",
                    "raw_completion": str(e),
                    "reference_code": problem_data.get('reference_code', ""),
                    "error": str(e)
                }
                generated_results.append(error_record) # Still append to keep track / for retry logic later
                write_jsonl(output_jsonl_path, [error_record], append=(processed_tasks > 0))
                processed_tasks += 1 # Count as processed even if error for now
            
            # Simple delay to be polite to local Ollama, can be adjusted/removed
            if not coder_model_name.startswith("gpt-"):
                time.sleep(0.1) 

    print(f"\nFinished processing. {len(generated_results)} records generated.")
    print(f"Results saved to {output_jsonl_path}")


def main():
    if not LANGCHAIN_AVAILABLE:
        print("Exiting due to missing Langchain components.")
        return

    parser = argparse.ArgumentParser(description="Generate code using LLMs based on prepared datasets.")
    parser.add_argument("--coder_model_name", type=str, required=True, help="Name/tag of the coder LLM (e.g., 'codellama:7b', 'gpt-4o-mini').")
    parser.add_argument("--dataset_name", type=str, required=True, choices=['humaneval', 'openeval'], help="Dataset to process ('humaneval' or 'openeval').")
    parser.add_argument("--plan_type", type=str, required=True, choices=['none', 'baseline', 'finetuned'], help="Type of plan to use for prompting ('none', 'baseline', 'finetuned').")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of code samples to generate per problem.")

    parser.add_argument("--prepared_data_dir", type=str, default="data", help="Directory containing the _prepared.jsonl files.")
    parser.add_argument("--output_results_dir", type=str, default="results", help="Base directory to save generation results.")
    parser.add_argument("--ollama_base_url", type=str, default=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"), help="Base URL for Ollama API.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for LLM generation. Use >0 for multiple diverse samples.")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum number of tokens to generate.")
    
    args = parser.parse_args()

    # Adjust temperature if generating more than 1 sample and current temp is 0
    if args.num_samples > 1 and args.temperature == 0.0:
        print("Warning: Generating multiple samples per problem, but temperature is 0.0. Setting temperature to 0.2 for diversity.")
        args.temperature = 0.2
        
    from dotenv import load_dotenv
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    generate_code_for_dataset(
        coder_model_name=args.coder_model_name,
        dataset_name=args.dataset_name,
        plan_type=args.plan_type,
        prepared_data_dir=args.prepared_data_dir,
        output_results_dir=args.output_results_dir,
        ollama_base_url=args.ollama_base_url,
        openai_api_key=openai_api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        num_samples_per_problem=args.num_samples
    )

if __name__ == "__main__":
    main() 