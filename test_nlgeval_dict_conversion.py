print("--- Starting NLGEval Internal-Like Dictionary Conversion Test ---")

hyps_input_list = ["this is a generated sentence", "this is another one"]
refs_input_list = [
    ["this is a reference sentence for the first one"],
    ["this is a reference for the second one", "this is an alternative reference for the second one"]
]

print(f"Length of hyps_input_list: {len(hyps_input_list)}")
print(f"Length of refs_input_list: {len(refs_input_list)}")

if len(hyps_input_list) != len(refs_input_list):
    print("Error: Input lists have different lengths. Test script is misconfigured.")
else:
    try:
        # Simulate the dictionary conversion as seen in nlgeval source
        # Assuming stripped_hyps would be the same as hyp item if no stripping/tokenization is done yet
        # and stripped_refs would be the same as ref item
        
        print("\nSimulating dictionary conversion for hypotheses...")
        # NLGEval actually stores tokenized hyps, but for length check, the item itself is just wrapped in a list usually.
        # Let's use the structure it implies for `hyps` which is a dict of lists of tokens/strings.
        # The key for `hyps` in nlgeval before assertion is `hyps = {idx: [stripped_hyp] ...}`
        hyps_dict = {idx: [hyp_item_text] for (idx, hyp_item_text) in enumerate(hyps_input_list)}
        print(f"Length of hyps_dict: {len(hyps_dict)}")
        if hyps_dict:
            print(f"Content of hyps_dict: {hyps_dict}")

        print("\nSimulating dictionary conversion for references...")
        # In nlgeval, stripped_refs is usually a list of tokenized reference strings for each original reference string.
        # So, refs_dict becomes dict mapping idx to list of lists of tokens/strings.
        # For this test, let's keep them as list of strings, mirroring `ref_list` structure where each item is a list of ref strings.
        refs_dict = {idx: ref_item_list_of_strings for (idx, ref_item_list_of_strings) in enumerate(refs_input_list)}
        print(f"Length of refs_dict: {len(refs_dict)}")
        if refs_dict:
            print(f"Content of refs_dict: {refs_dict}")

        print("\nChecking assertion condition...")
        if len(hyps_dict) == len(refs_dict):
            print("Success: len(hyps_dict) == len(refs_dict)")
        else:
            print(f"Failure: len(hyps_dict) [{len(hyps_dict)}] != len(refs_dict) [{len(refs_dict)}]")
            print("This suggests the dictionary conversion itself might be the issue IF nlgeval does something similar.")

    except Exception as e:
        print(f"\nError during dictionary conversion simulation: {type(e).__name__}: {e}")

print("--- NLGEval Internal-Like Dictionary Conversion Test Finished ---") 