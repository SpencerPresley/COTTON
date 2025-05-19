from nlgeval import NLGEval

print("--- Starting Minimal NLGEval Test ---")

hyps_test_list = ["this is a generated sentence", "this is another one"]
# Each item in ref_list corresponds to a hypothesis and is a list of reference strings for that hypothesis.
refs_test_list = [
    ["this is a reference sentence for the first one"],
    ["this is a reference for the second one", "this is an alternative reference for the second one"]
]

print(f"Length of hyps_test_list: {len(hyps_test_list)}")
print(f"Length of refs_test_list: {len(refs_test_list)}")

if len(hyps_test_list) != len(refs_test_list):
    print("Error: Test lists have different lengths before calling NLGEval. This test script is misconfigured.")
else:
    print("Initializing NLGEval for minimal test...")
    nlgeval_minimal = NLGEval(no_skipthoughts=True, no_glove=True)
    
    print("Computing metrics with minimal test data...")
    try:
        metrics = nlgeval_minimal.compute_metrics(ref_list=refs_test_list, hyp_list=hyps_test_list)
        print("\nMinimal NLGEval Test - Metrics Computed Successfully:")
        for metric, score in metrics.items():
            print(f"  {metric}: {score}")
    except AssertionError as e:
        print(f"\nMinimal NLGEval Test - AssertionError from nlgeval:")
        print(f"  Error: {e}")
        print(f"  Length of hyps passed to compute_metrics: {len(hyps_test_list)}")
        print(f"  Length of refs passed to compute_metrics: {len(refs_test_list)}")
    except Exception as e:
        print(f"\nMinimal NLGEval Test - Other error from nlgeval:")
        print(f"  Error: {type(e).__name__}: {e}")

print("--- Minimal NLGEval Test Finished ---") 