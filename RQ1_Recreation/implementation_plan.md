# Step-by-Step Implementation Plan for RQ1 Recreation

**Phase 0: Prerequisites and Setup**

1. **Create Workspace:**
    * Ensure a new directory `COTTON/RQ1_Recreation/` is created. All new scripts and results for this phase will reside here.
    * A subdirectory `COTTON/RQ1_Recreation/results/` will be used for storing generated code outputs, organized by model and dataset.
    * Another subdirectory `COTTON/RQ1_Recreation/data/` will store the prepared datasets.

2. **Environment:**
    * Ensure `langchain`, `langchain-ollama`, `langchain-openai`, `pandas`, and `openai` Python libraries are installed and accessible.
    * Ollama service should be running and the required models pulled.
    * OpenAI API key should be available (e.g., as an environment variable).

**Phase 1: Data Preparation**

*Objective: Create consolidated JSONL datasets for HumanEval and OpenEval that include the problem, reference code, reference plan, and the pre-generated plans from both baseline and fine-tuned TinyLlama models.*

1. **Script: `COTTON/RQ1_Recreation/prepare_datasets.py`**
    * **Inputs:**
        * `COTTON/dataset/humaneval.csv` (cols: `src`, `tgt`)
        * `COTTON/dataset/openeval.csv` (cols: `src`, `tgt`)
        * Baseline TinyLlama plans:
            * `COTTON/ollama_baseline_results/test_humaneval/predictions.csv`
            * `COTTON/ollama_baseline_results/test_openeval/predictions.csv`
        * Fine-tuned TinyLlama plans:
            * `COTTON/save_model_llama3.2_fast_eval/test_humaneval/predictions.csv`
            * `COTTON/save_model_llama3.2_fast_eval/test_openeval/predictions.csv`
        * **Crucial External Input (to be confirmed with user):** Paths to the *full* HumanEval and OpenEval datasets (ideally JSONL format, like the original `HumanEval.jsonl.gz`) which contain `task_id`, `prompt` (this should align with the `src` column of the CSVs), and `canonical_solution` (the reference code).
            * *Contingency:* If full JSONL is unavailable, I will proceed using `src` from the CSVs as `problem_prompt` and will omit `task_id` and `reference_code` from the prepared data, noting this limitation for evaluation.
    * **Outputs:**
        * `COTTON/RQ1_Recreation/data/humaneval_prepared.jsonl`
        * `COTTON/RQ1_Recreation/data/openeval_prepared.jsonl`
    * **Logic (for each dataset - HumanEval/OpenEval):**
        1. Load the main dataset CSV (e.g., `COTTON/dataset/humaneval.csv`). Let\'s call its columns `problem_src_csv` and `reference_plan_csv`.
        2. Load the baseline plans CSV.
        3. Load the fine-tuned plans CSV.
        4. Load the full JSONL dataset (e.g., `HumanEval.jsonl`).
        5. Iterate from `i = 0` to `len(main_dataset_csv) - 1`. For each row `i`:
            * `task_id_from_jsonl = full_jsonl[i][\'task_id\']`
            * `problem_prompt_from_jsonl = full_jsonl[i][\'prompt\']`
            * `reference_code_from_jsonl = full_jsonl[i][\'canonical_solution\']`
            * `problem_prompt_from_csv = main_dataset_csv[i][\'src\']`
            * `reference_plan_from_csv = main_dataset_csv[i][\'tgt\']`
            * `baseline_plan = baseline_plans_csv[i][0]` (assuming single column CSV)
            * `finetuned_plan = finetuned_plans_csv[i][0]` (assuming single column CSV)
            * **Verification:** Assert that `problem_prompt_from_csv` is identical or very similar to `problem_prompt_from_jsonl` to ensure correct alignment. If not, this is a critical issue.
            * Construct a JSON object:

                ```json
                {
                    "task_id": task_id_from_jsonl,
                    "problem_prompt": problem_prompt_from_csv, // or from_jsonl if confirmed identical
                    "reference_plan": reference_plan_from_csv,
                    "baseline_tinyllama_plan": baseline_plan,
                    "finetuned_tinyllama_plan": finetuned_plan,
                    "reference_code": reference_code_from_jsonl
                }
                ```

            * Append this JSON object as a new line to the output JSONL file (e.g., `humaneval_prepared.jsonl`).
    * This script will be run once to generate the prepared datasets.

**Phase 2: Code Generation**

*Objective: Generate code solutions using various coder LLMs, under different prompting conditions (no plan, baseline plan, fine-tuned plan).*

1. **Script: `COTTON/RQ1_Recreation/generate_code.py`**
    * **Libraries:** `argparse`, `pandas`, `json`, `langchain_ollama.llms.OllamaLLM`, `langchain_openai.ChatOpenAI`, `langchain_core.prompts.PromptTemplate`, `os`.
    * **Command-line Arguments:**
        * `--coder_model_name` (string, required): The name/tag of the coder LLM (e.g., `codellama:7b-instruct-q4_0`, `starcoder2:3b`, `qwen2:7b-instruct-q4_0`, `gpt-4o-mini`).
        * `--dataset_name` (string, required, choices: `humaneval`, `openeval`): Which dataset to process.
        * `--plan_type` (string, required, choices: `none`, `baseline`, `finetuned`): Which type of plan to include in the prompt.
        * `--output_dir` (string, default: `COTTON/RQ1_Recreation/results/`): Base directory for output files.
        * `--ollama_base_url` (string, default: `http://localhost:11434`): For Ollama models.
        * `--openai_api_key` (string, default: `os.environ.get("OPENAI_API_KEY")`): For OpenAI models.
        * `--temperature` (float, default: 0.0 for reproducibility, but can be 0.2 as in RQ1).
        * `--max_tokens` (int, default: 1024).
    * **Logic:**
        1. Parse arguments.
        2. Determine the input prepared dataset path: `COTTON/RQ1_Recreation/data/{args.dataset_name}_prepared.jsonl`.
        3. Initialize the LLM:
            * If `args.coder_model_name` starts with "gpt-", initialize `ChatOpenAI`.
            * Else, initialize `OllamaLLM` with `model=args.coder_model_name`, `base_url=args.ollama_base_url`, `temperature=args.temperature`.
        4. Load the prepared dataset (JSONL file) line by line.
        5. Define prompt templates:
            * **`prompt_template_with_plan`**:

                ```
                You are an expert Python programmer.
                Based on the following problem description and implementation plan, please write the Python code solution.
                Only provide the complete Python code block. Do not include any explanations or surrounding text.

                ### Problem:
                {problem_prompt}

                ### Implementation Plan:
                {plan_steps}

                ### Python Code:
                ```

            * **`prompt_template_no_plan`**:

                ```
                You are an expert Python programmer.
                Based on the following problem description, please write the Python code solution.
                Only provide the complete Python code block. Do not include any explanations or surrounding text.

                ### Problem:
                {problem_prompt}

                ### Python Code:
                ```

        6. Prepare output file path: `args.output_dir/{args.coder_model_name.replace(":", "_")}/{args.dataset_name}_with_{args.plan_type}_plan_generations.jsonl`. Ensure the directory exists.
        7. Open the output file for writing.
        8. For each `record` in the loaded prepared dataset:
            * `task_id = record[\'task_id\']`
            * `problem_prompt = record[\'problem_prompt\']`
            * `reference_code = record[\'reference_code\']`
            * `plan_to_use = None`
            * If `args.plan_type == \'baseline\'`: `plan_to_use = record[\'baseline_tinyllama_plan\']`
            * Else if `args.plan_type == \'finetuned\'`: `plan_to_use = record[\'finetuned_tinyllama_plan\']`

            * If `plan_to_use` is not `None`:
                * `current_prompt = PromptTemplate.from_template(prompt_template_with_plan).format(problem_prompt=problem_prompt, plan_steps=plan_to_use)`
            * Else (`args.plan_type == \'none\'`):
                * `current_prompt = PromptTemplate.from_template(prompt_template_no_plan).format(problem_prompt=problem_prompt)`

            * Invoke the LLM: `generated_code = llm.invoke(current_prompt)` (Add error handling/retry if specified).
            * Post-process `generated_code`: Extract only the Python code block if the LLM adds extra text (e.g., using regex to find ```python ...```).
            * Store result:

                ```json
                {
                    "task_id": task_id,
                    "problem_prompt": problem_prompt,
                    "plan_type_used": args.plan_type,
                    "plan_steps_used": plan_to_use if plan_to_use else "",
                    "generated_code": generated_code,
                    "reference_code": reference_code
                }
                ```

            * Write this JSON object to the output file as a new line.
            * Print progress (e.g., `Processed task {task_id}`).
        9. Close output file.

**Phase 3: Running Experiments**

*Objective: Systematically run `generate_code.py` for all combinations of coder models, datasets, and plan types.*

1. **Coder Models List & Parameters:**
    * **CodeLlama (Ollama):** (e.g., `codellama:7b-instruct-q4_0`, `codellama:13b-instruct-q4_0`)
    * **Qwen2.5-coder (Ollama):** (e.g., `qwen2:0.5b-instruct-q4_0`, `qwen2:1.5b-instruct-q4_0`, `qwen2:7b-instruct-q4_0` - *confirm exact tags for coder variants if different from base instruct*).
    * **Starcoder2 (Ollama):** (e.g., `starcoder2:3b-instruct-q4_0`, `starcoder2:7b-instruct-q4_0`)
    * **OpenAI:** `gpt-4o-mini` (or other specified model).
2. **Execution Loop (Conceptual - can be a shell script or manual execution):**

    ```
    for coder_model in [list of coder models]:
        for dataset_name in ["humaneval", "openeval"]:
            for plan_type in ["none", "baseline", "finetuned"]:
                python COTTON/RQ1_Recreation/generate_code.py \
                    --coder_model_name "{coder_model}" \
                    --dataset_name "{dataset_name}" \
                    --plan_type "{plan_type}" \
                    # Add --ollama_base_url or --openai_api_key as needed
    ```

**Phase 4: Evaluation (Future Step - Not part of this plan\'s implementation)**

* The generated JSONL files from Phase 2 will be in a format suitable for standard code evaluation harnesses (e.g., `bigcode-evaluation-harness`, custom scripts for `pass@k`).
* This plan focuses on setting up the data and generating the code.

---

This detailed plan should provide a clear path forward. Before I proceed with generating tool calls or asking for file modifications, I need your input on the following:

**Questions & Context Needed:**

1. **Location of Full HumanEval/OpenEval Datasets:** Could you please provide the paths to the full HumanEval and OpenEval datasets (preferably JSONL files) that include `task_id`, `prompt` (the problem description, which should align with the `src` column in your `COTTON/dataset/*.csv` files), and `canonical_solution` (the reference Python code)? This is essential for the `prepare_datasets.py` script.
2. **Exact Ollama Model Names/Tags:** For the coder models like Qwen2.5-coder and Starcoder2, could you specify the exact model tags you intend to use with Ollama (e.g., `qwen2:7b-instruct-q4_0`, `starcoder2:3b-instruct` or similar)? Sometimes coder-specific variants have different tags than base or general instruct models.
3. **OpenAI Model Choice:** `gpt-4o-mini` is noted. Is this your final choice, or would you like this to be a configurable parameter?
4. **Output Format Confirmation:** The `generate_code.py` script will output JSONL files with fields: `task_id`, `problem_prompt`, `plan_type_used`, `plan_steps_used`, `generated_code`, `reference_code`. Is this acceptable for your downstream evaluation?
5. **Error Handling for LLM Calls:** Should the `generate_code.py` script include specific error handling (e.g., retries on API failures, logging errors for specific tasks) when calling Ollama or OpenAI models?
6. **Prompting Nuances for Code Generation:** I\'ve drafted generic code generation prompts. Do you have any specific instructions, few-shot examples, or system messages you\'d like to incorporate when prompting the coder LLMs, especially when a plan is provided?
7. **Cleaning Generated Plans:** As noted, the plans from the baseline TinyLlama (via `generate_plans_ollama.py`) might contain extra text or even code. Should the `prepare_datasets.py` script attempt to clean/extract only the step-by-step plan part from these `predictions.csv` files, or should they be used as-is? The fine-tuned plans seem cleaner.
