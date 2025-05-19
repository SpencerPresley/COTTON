# Changelog for Llama 3.2 Integration

This document outlines the changes made to integrate the `meta-llama/Llama-3.2-1B-Instruct` model into this repository.

## Key Goals

- Replace the existing Llama model with `meta-llama/Llama-3.2-1B-Instruct`.
- Enable fine-tuning of the new model with 4-bit quantization.
- Ensure inference (prediction) works correctly with the new model.
- Ensure the script runs on GPU and errors out if GPU is not available.

## File-Specific Changes

### 1. `COTTON/LLAMA_Model.py`

- **Imports:**
  - Removed unused import `prepare_model_for_int8_training` from `peft` to resolve `ImportError` with `peft>=0.10.0` (user is on `0.15.2`).
  - Changed `AdamW` import from `transformers` to `torch.optim` to resolve `ImportError` with newer `transformers` versions.
- **Model Loading (`get_model_tokenizer` method):**
  - Changed `LlamaForCausalLM.from_pretrained` to `AutoModelForCausalLM.from_pretrained`.
  - Changed `CodeLlamaTokenizer.from_pretrained` to `AutoTokenizer.from_pretrained`.
  - Added imports for `AutoModelForCausalLM`, `AutoTokenizer`, and `BitsAndBytesConfig`.
  - The `base_model` path will now be `meta-llama/Llama-3.2-1B-Instruct` (passed during class instantiation).
  - **Enabled 4-bit Quantization:**
    - Defined a `BitsAndBytesConfig` for 4-bit nf4 quantization with `bfloat16` compute type.
    - Passed this `quantization_config` to `AutoModelForCausalLM.from_pretrained`.
    - Removed `torch_dtype=torch.bfloat16` from `from_pretrained` as it's handled by the quantization config.

- **PEFT Preparation (`__init__` method):**
  - Imported `prepare_model_for_kbit_training` from `peft`.
  - Called `self.model = prepare_model_for_kbit_training(self.model)` after model loading to prepare the quantized model for LoRA.

- **GPU Check (`__init__` method):**
  - Added a check after `self.device` is set.
  - If `self.device.type` is `'cpu'`, a `RuntimeError` is raised to prevent running without a GPU.
  - Prints the device being used if a GPU is found.

- **Prediction (`predict` method):**
  - Modified to work with the new chat-based prompt format.
  - It now calls `cot_prompt_pre` (from `custom_datasets.py`) which returns a list of messages.
  - Uses `tokenizer.apply_chat_template` to format and tokenize the messages for the model.
  - `max_length` for the prompt tokenization is set using `self.source_len`.
  - The input to `model.generate` is now `input_ids` obtained from `apply_chat_template`.
  - Kept the logic to slice off the prompt tokens from the generated output based on the length of `input_ids`.

- **LoRA Configuration (`load_adapter_config` and `find_all_linear_names`):**
  - No code changes made yet to the logic of `find_all_linear_names`.
  - **To-Do:** Investigate if `target_modules` for LoRA need to be different for `meta-llama/Llama-3.2-1B-Instruct`. The current implementation targets all `nn.Linear` layers (excluding `lm_head`). This might need to be updated based on Llama 3.2's architecture or specific recommendations for optimal LoRA fine-tuning. `prepare_model_for_kbit_training` should generally handle LoRA application to quantized layers correctly.

### 2. `COTTON/train.py`

- **Model Instantiation:**
  - Updated the `base_model_path` argument to `"meta-llama/Llama-3.2-1B-Instruct"` in both places where `LLAMASeq2Seq` is initialized (for training and for testing).
- **"Fast Run" Configuration (to meet time constraints):**
  - Adjusted training parameters for a quicker run:
    - `NUM_EPOCHS` set to `3`.
    - `BATCH_SIZE` set to `4` (for training and evaluation).
    - `OUTPUT_DIR` changed to `'save_model_llama3.2_fast_eval/'`.
    - `DO_EVAL_BLEU` kept as `True` per user request, but noted that this significantly impacts epoch time.
  - Advised manual dataset subsetting for further speed-up if original datasets are very large.
  - **Added `nrows=1000` for `GPTDataset` during training if `IS_FAST_RUN` is `True`, to use only the first 1000 rows of the training data.**

### 3. `COTTON/custom_datasets.py`

- **Prompt Formatting for Inference (`cot_prompt_pre` function):**
  - Changed from creating a single formatted string to returning a list of dictionaries representing chat messages (system and user roles).
  - Example: `[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]`
  - Added a default system prompt: `"You are a helpful assistant that provides implementation ideas for code."`

- **Data Preparation for Fine-tuning (`GPTDataset` class):**
  - **Implemented (Major Changes):**
    - **`__init__` Method:**
      - Now accepts an optional `system_prompt_content` argument to set a system message for all training examples.
      - **Added an optional `nrows` parameter to limit the number of rows read by `pd.read_csv` or `pd.read_json`.**
      - Stores `tokenizer` and `cutoff_len` as instance attributes.
      - The `source_len` parameter's role is diminished due to `cutoff_len` applying to the whole chat sequence.
      - The call to `tokenize_prompt` in the `__init__` loop now passes only `src` and `tgt`.
    - **`tokenize_prompt(self, src, tgt)` Method (Complete Overhaul):**
        1. **Message Construction:** Creates a `messages` list:
            - Optional system message (if `self.system_prompt_content` is provided).
            - User message with `src`.
            - Assistant message with `tgt`.
        2. **Tokenization for `input_ids`:**
            - Uses `self.tokenizer.apply_chat_template(messages, max_length=self.cutoff_len, truncation=True, add_generation_prompt=False, return_tensors=None)` to get `input_ids`. `add_generation_prompt=False` is used because the assistant's reply (`tgt`) is part of the `messages`.
        3. **Label Creation and Masking:**
            - `labels` are initialized as a copy of `input_ids` (as a Python list).
            - To determine the prompt portion to mask:
                - A temporary `prompt_messages_for_masking` list (system + user message) is created.
                - This list is tokenized using `self.tokenizer.apply_chat_template(..., add_generation_prompt=True)`. The `add_generation_prompt=True` is crucial here as it includes all template tokens leading up to where the assistant's response would begin.
                - The length of these `tokenized_prompt_part_ids` is `prompt_length`.
            - Tokens in `labels` from index `0` up to `prompt_length - 1` are set to `-100`.
            - Includes an `assert` to ensure `input_ids` and `labels` have the same length.
    - **Removed Old `tokenize` Helper Method:** This method is no longer needed.
  - The `GPTDataset` class is now adapted to prepare data in the chat format suitable for fine-tuning `meta-llama/Llama-3.2-1B-Instruct`.

### 4. Dependencies (e.g., `requirements.txt`)

- No new file created yet.
- **User Confirmed Done:** User has confirmed dependencies are installed (ensure `bitsandbytes` is included and compatible).
- **Recommendation:** Ensure `transformers` library is version `4.43.0` or newer to support Llama 3.2 chat templates effectively.
- **Note:** The user previously had an issue installing `nlgeval`. This should be resolved if evaluation with it is still planned.

### 5. `COTTON/readme.md`

- **To-Do:** Update the `readme.md` to reflect the change to using `meta-llama/Llama-3.2-1B-Instruct`, mention 4-bit quantization, and briefly cover any new setup or usage instructions if applicable.

## Current Status

- Model loading and the `base_model_path` are updated.
- **4-bit quantization is enabled** for model loading.
- **Script will error if no GPU is detected.**
- **Inference (`predict` method) has been adapted for the Llama 3.2 chat format.**
- **Fine-tuning data preparation (`GPTDataset`) has been significantly updated to use chat templates and appropriate label masking for Llama 3.2.**
- LoRA target modules might need review (though `prepare_model_for_kbit_training` helps).

## Next Steps Before Running

1. (Optional but Recommended) Review LoRA `target_modules` for Llama 3.2 (see `find_all_linear_names` and printed output during training startup).
2. Update `readme.md`.
3. **Test fine-tuning and inference thoroughly.** Remember to start with a small number of epochs and potentially disable BLEU evaluation for the very first test run to quickly verify the pipeline.
