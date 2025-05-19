import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA is not available to PyTorch.")

# Check bitsandbytes path (might be useful for advanced debugging, but primary issue is likely drivers)
try:
    import bitsandbytes
    print(f"bitsandbytes version: {getattr(bitsandbytes, '__version__', 'unknown')}")
    print(f"bitsandbytes path: {bitsandbytes.__file__}")
except ImportError:
    print("bitsandbytes is not imported.") 