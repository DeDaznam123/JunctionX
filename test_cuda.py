#!/usr/bin/env python3
"""Test script to verify CUDA setup for faster-whisper"""

print("Testing CUDA setup for faster-whisper...")
print("-" * 50)

try:
    from faster_whisper import WhisperModel
    print("✓ faster-whisper imported successfully")

    print("\nAttempting to initialize model with CUDA...")
    model = WhisperModel("tiny", device="cuda", compute_type="float16")
    print("✓ SUCCESS! CUDA is working with faster-whisper!")
    print(f"✓ Model loaded on GPU")

except Exception as e:
    print(f"✗ CUDA initialization failed:")
    print(f"  Error: {e}")
    print("\nTrying CPU fallback...")
    try:
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        print("✓ CPU fallback successful")
    except Exception as cpu_err:
        print(f"✗ CPU also failed: {cpu_err}")

print("-" * 50)
print("Test complete!")
