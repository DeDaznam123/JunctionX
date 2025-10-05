#!/usr/bin/env python3
"""Test script to verify BART can use GPU"""

print("Testing GPU setup for BART classifier...")
print("-" * 50)

# 1. Check PyTorch and CUDA
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ GPU count: {torch.cuda.device_count()}")
        print(f"✓ GPU name: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("✗ CUDA is NOT available!")
        print("  You need to install PyTorch with CUDA support:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

except Exception as e:
    print(f"✗ PyTorch error: {e}")

print("-" * 50)

# 2. Test BART model loading
try:
    from transformers import pipeline
    print("\n✓ Transformers imported successfully")

    if torch.cuda.is_available():
        print("\nLoading BART model on GPU...")
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0  # GPU
        )
        print("✓ SUCCESS! BART loaded on GPU!")

        # Quick inference test
        result = classifier("This is a test", candidate_labels=["positive", "negative"])
        print(f"✓ Inference test passed: {result['labels'][0]} ({result['scores'][0]:.2f})")
    else:
        print("\n⚠ CUDA not available, loading on CPU...")
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1  # CPU
        )
        print("✓ BART loaded on CPU (fallback)")

except Exception as e:
    print(f"✗ Model loading failed: {e}")

print("-" * 50)
print("Test complete!")
