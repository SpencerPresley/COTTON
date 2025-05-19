import argparse
from nlp2 import set_seed
from LLAMA_Model import LLAMASeq2Seq

def setup_args():
    parser = argparse.ArgumentParser(description="Train and/or evaluate LLAMA model.")
    parser.add_argument("--base_model_path", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Path to the base LLaMA model.")
    parser.add_argument("--output_dir", type=str, default="save_model_llama3.2_fast_eval/", help="Directory to save model checkpoints and test results.")
    parser.add_argument("--train_filename", type=str, default="dataset/train.csv", help="Path to the training data CSV file.")
    parser.add_argument("--eval_filename", type=str, default="dataset/valid.csv", help="Path to the evaluation data CSV file.")
    parser.add_argument("--humaneval_filename", type=str, default="dataset/humaneval.csv", help="Path to the HumanEval test data CSV file.")
    parser.add_argument("--openeval_filename", type=str, default="dataset/openeval.csv", help="Path to the OpenEval test data CSV file.")
    
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and evaluation.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for training.")
    parser.add_argument("--early_stop_patience", type=int, default=3, help="Patience for early stopping based on BLEU score.")
    
    parser.add_argument("--source_len", type=int, default=256, help="Maximum source sequence length.")
    parser.add_argument("--cutoff_len", type=int, default=512, help="Maximum total sequence length (source + target).")
    parser.add_argument("--adapter_type", type=str, default="lora", help="Adapter type for PEFT (e.g., lora).")
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--train_nrows", type=int, default=None, help="Number of rows to read from training data (for fast runs). Set to 1000 for IS_FAST_RUN=True equivalent.")
    parser.add_argument("--is_fast_run_train_nrows", type=int, default=1000, help="Number of rows for training if --use_fast_run is active (legacy IS_FAST_RUN).")
    parser.add_argument("--use_fast_run", action="store_true", help="Use a subset of training data (defined by --is_fast_run_train_nrows).")


    parser.add_argument("--skip_train", action="store_true", help="Skip the training phase and only run testing.")
    parser.add_argument("--no_eval_bleu", action="store_true", help="Disable BLEU evaluation during training.")
    parser.add_argument("--add_eos_token", action="store_true", help="Add EOS token during model/tokenizer setup (defaults to False).")

    return parser.parse_args()

def initialize_model(base_model_path, load_adapter_path, add_eos_token, adapter, source_len, cutoff_len):
    print(f"Initializing model. Base: {base_model_path}, Adapter: {load_adapter_path if load_adapter_path else 'None'}")
    return LLAMASeq2Seq(
        base_model_path=base_model_path,
        add_eos_token=add_eos_token,
        adapter=adapter,
        load_adapter_path=load_adapter_path if load_adapter_path else "None", # Ensure "None" string if path is None
        source_len=source_len,
        cutoff_len=cutoff_len
    )

def run_training_phase(model, args):
    print("--- Starting Training Phase ---")
    
    train_nrows_to_use = args.train_nrows
    if args.use_fast_run and args.train_nrows is None: # If use_fast_run is on and train_nrows not explicitly set
        train_nrows_to_use = args.is_fast_run_train_nrows

    model.train(
        train_filename=args.train_filename,
        train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        early_stop=args.early_stop_patience,
        do_eval=True, # Evaluation during training is generally good
        eval_filename=args.eval_filename,
        eval_batch_size=args.batch_size,
        output_dir=args.output_dir,
        do_eval_bleu=not args.no_eval_bleu,
        train_nrows=train_nrows_to_use
    )
    print("--- Training Phase Finished ---")

def run_testing_phase(model, test_filename, output_dir_test_specific):
    print(f"--- Running Test on {test_filename} ---")
    model.test(filename=test_filename, output_dir=output_dir_test_specific)
    print(f"--- Test on {test_filename} Finished ---")

def main():
    args = setup_args()
    set_seed(args.seed)

    if not args.skip_train:
        # Initialize model for training (no pre-trained adapter)
        training_model = initialize_model(
            base_model_path=args.base_model_path,
            load_adapter_path=None, # No adapter to load for initial training
            add_eos_token=args.add_eos_token,
            adapter=args.adapter_type,
            source_len=args.source_len,
            cutoff_len=args.cutoff_len
        )
        run_training_phase(training_model, args)

    # Initialize model for testing (loading the best checkpoint)
    # Ensure the checkpoint name 'checkpoint-best-bleu' is correct if BLEU is used.
    # If BLEU is off, the best model might be saved differently (e.g., 'checkpoint-best-loss').
    # For simplicity, we assume 'checkpoint-best-bleu' if BLEU eval was on.
    # If BLEU was off, this path might need adjustment or the script might save a default best model.
    best_model_path = f"{args.output_dir}/checkpoint-best-bleu"
    if args.no_eval_bleu and not args.skip_train : # If BLEU was off during training
         print(f"Warning: BLEU evaluation was off during training. Attempting to load best model from {best_model_path}, but it might be based on loss or not exist if training didn't save optimally without BLEU.")
         # Potentially, LLAMA_Model.py saves a 'checkpoint-best-loss' or similar. For now, we stick to this.
         # Or, if training was skipped, the user must ensure the checkpoint exists.
    
    print(f"Attempting to load model for testing from: {best_model_path}")
    testing_model = initialize_model(
        base_model_path=args.base_model_path,
        load_adapter_path=best_model_path,
        add_eos_token=args.add_eos_token,
        adapter=args.adapter_type,
        source_len=args.source_len,
        cutoff_len=args.cutoff_len
    )

    # Run tests
    run_testing_phase(testing_model, args.humaneval_filename, f'{args.output_dir}/test_humaneval/')
    run_testing_phase(testing_model, args.openeval_filename, f'{args.output_dir}/test_openeval/')

if __name__ == "__main__":
    main()