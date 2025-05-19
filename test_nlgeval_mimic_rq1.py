from nlgeval import NLGEval

print("--- Starting NLGEval Test (Mimicking RQ1 compute_metrics call) ---")

# Simulating flat lists as read from CSV in RQ1
flat_hyp_list = ["this is a generated sentence", "this is another one"]
flat_ref_list = ["this is a reference sentence for the first one", "this is a reference for the second one"] # Note: NOT list of lists yet

print(f"Length of flat_hyp_list: {len(flat_hyp_list)}")
print(f"Length of flat_ref_list: {len(flat_ref_list)}")

# This is how RQ1's compute_metrics calls nlgeval.compute_metrics:
# ref_list_arg_to_nlgeval = [flat_ref_list]
# hyp_list_arg_to_nlgeval = flat_hyp_list
# This means len(ref_list_arg_to_nlgeval) would be 1, and len(hyp_list_arg_to_nlgeval) would be 2.

print("\nInitializing NLGEval...")
# nlgeval_test = NLGEval(no_skipthoughts=True, no_glove=True)
# Initialize NLGEval disabling SPICE and METEOR to avoid Java/external dependency issues for this test
nlgeval_test = NLGEval(no_skipthoughts=True, no_glove=True, metrics_to_omit=['SPICE'])

print("\nComputing metrics with RQ1-style call structure (passing [flat_ref_list])...")
# This is the potentially problematic call structure from RQ1's compute_metrics
# It passes ref_list=[flat_ref_list], so the outer list has only 1 element.
# If flat_hyp_list has >1 element, this should cause assert len(refs) == len(hyps) to fail in nlgeval.
passed_ref_list_arg = [flat_ref_list]
passed_hyp_list_arg = flat_hyp_list

try:
    print(f"Calling nlgeval with len(ref_list_arg)={len(passed_ref_list_arg)}, len(hyp_list_arg)={len(passed_hyp_list_arg)}")
    metrics = nlgeval_test.compute_metrics(ref_list=passed_ref_list_arg, hyp_list=passed_hyp_list_arg)
    print("\nMetrics Computed Successfully (RQ1 style call):")
    for metric, score in metrics.items():
        print(f"  {metric}: {score}")
except AssertionError as e:
    print(f"\nAssertionError from nlgeval (RQ1 style call):")
    print(f"  Error: {e}")
    print(f"  Length of hyps effectively passed to nlgeval: {len(passed_hyp_list_arg)}")
    print(f"  Length of refs (outer list) effectively passed to nlgeval: {len(passed_ref_list_arg)}") 
except Exception as e:
    print(f"\nOther error from nlgeval (RQ1 style call):")
    print(f"  Error: {type(e).__name__}: {e}")

print("\n--- Now trying the standard paired call structure for nlgeval ---")
# Standard way: list of hypotheses, and a list of lists of references (one list of refs per hyp)
# This is what evaluate_plans.py does.
paired_refs_list = [[r] for r in flat_ref_list]

print(f"Length of flat_hyp_list (for standard call): {len(flat_hyp_list)}")
print(f"Length of paired_refs_list (for standard call): {len(paired_refs_list)}")

try:
    print(f"Calling nlgeval with len(ref_list_arg)={len(paired_refs_list)}, len(hyp_list_arg)={len(flat_hyp_list)}")
    metrics_paired = nlgeval_test.compute_metrics(ref_list=paired_refs_list, hyp_list=flat_hyp_list)
    print("\nMetrics Computed Successfully (Standard Paired Call):")
    for metric, score in metrics_paired.items():
        print(f"  {metric}: {score}")
except AssertionError as e:
    print(f"\nAssertionError from nlgeval (Standard Paired Call):")
    print(f"  Error: {e}")
    print(f"  Length of hyps effectively passed to nlgeval: {len(flat_hyp_list)}")
    print(f"  Length of refs (outer list) effectively passed to nlgeval: {len(paired_refs_list)}")
except Exception as e:
    print(f"\nOther error from nlgeval (Standard Paired Call):")
    print(f"  Error: {type(e).__name__}: {e}")


print("--- NLGEval Test (Mimicking RQ1 compute_metrics call) Finished ---") 