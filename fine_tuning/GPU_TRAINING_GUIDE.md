# NVIDIA Blackwell GPU Training Guide

## 🚀 Quick Setup

Your system has an **NVIDIA Blackwell GPU**, but currently PyTorch is using the CPU-only version. Follow these steps to enable GPU acceleration.

## Step 1: Install CUDA-Enabled PyTorch

Run the GPU setup script:
```powershell
cd fine_tuning
.\setup_gpu.ps1
```

This will:
1. Uninstall CPU-only PyTorch
2. Install CUDA 12.4-enabled PyTorch (optimized for Blackwell)
3. Verify GPU detection
4. Install optimization libraries

### Manual Installation (Alternative)

If the script doesn't work, install manually:
```powershell
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## Step 2: Verify GPU Setup

After installation, verify GPU is detected:
```powershell
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output:
```
CUDA Available: True
GPU: NVIDIA Blackwell [Your GPU Model]
```

## Step 3: Train with GPU

### Recommended Command (BF16 - Best for Blackwell)
```bash
python train_model.py \
  --data_dir ./data \
  --output_dir ./outputs \
  --bf16 \
  --batch_size 32 \
  --num_epochs 3
```

### Alternative (FP16 - Also Fast)
```bash
python train_model.py \
  --data_dir ./data \
  --output_dir ./outputs \
  --fp16 \
  --batch_size 32 \
  --num_epochs 3
```

### No Mixed Precision (Slowest, Most Precise)
```bash
python train_model.py \
  --data_dir ./data \
  --output_dir ./outputs \
  --batch_size 16
```

## 🎯 Blackwell-Specific Optimizations

### What is Blackwell?
NVIDIA Blackwell is the latest GPU architecture (2024-2025) with:
- **Compute Capability 10.x**
- Enhanced BF16/FP16 performance
- Higher memory bandwidth
- Advanced tensor cores

### Why Use BF16?
**BF16 (BFloat16)** is the recommended precision for Blackwell GPUs:
- ✅ **Faster training** (2-3x speedup vs FP32)
- ✅ **Better numerical stability** than FP16
- ✅ **Native support** on Blackwell architecture
- ✅ **Lower memory usage** (50% less than FP32)

### Batch Size Optimization

| GPU Memory | Recommended Batch Size | Mixed Precision |
|------------|------------------------|-----------------|
| 8-12 GB    | 16-24                 | --bf16          |
| 16-24 GB   | 32-48                 | --bf16          |
| 32-48 GB   | 64-96                 | --bf16          |
| 80+ GB     | 128+                  | --bf16          |

**Note**: Start with smaller batch sizes and increase until you hit OOM (Out of Memory) errors.

## 🔧 Advanced GPU Settings

### Enable in Training Script

The training script now includes several GPU optimizations:

1. **BF16 Mixed Precision** (`--bf16`)
   - Best performance on Blackwell
   - Better numerical stability than FP16

2. **Automatic GPU Detection**
   - Displays GPU info at training start
   - Shows memory usage and compute capability

3. **Dataloader Optimizations**
   - Pin memory for faster GPU transfers
   - Multi-worker data loading (2 workers)

4. **Fused AdamW Optimizer**
   - Uses PyTorch's optimized AdamW
   - Better performance on CUDA

### Monitoring GPU Usage

During training, monitor GPU with:
```powershell
# In a separate terminal
nvidia-smi -l 1
```

This shows:
- GPU utilization (should be 90-100% during training)
- Memory usage
- Temperature
- Power consumption

### Expected Performance

**Training Speed Comparison:**
| Configuration | Speed (samples/sec) | Relative Speed |
|--------------|---------------------|----------------|
| CPU Only     | ~2-5                | 1x (baseline)  |
| GPU FP32     | ~80-120             | 20-40x         |
| GPU FP16     | ~160-240            | 40-80x         |
| GPU BF16     | ~160-250            | 40-85x         |

**Memory Usage:**
| Precision | Model Size | Peak Memory |
|-----------|------------|-------------|
| FP32      | ~500 MB    | ~3-4 GB     |
| FP16      | ~250 MB    | ~2-3 GB     |
| BF16      | ~250 MB    | ~2-3 GB     |

## 🐛 Troubleshooting

### Issue: "CUDA out of memory"
**Solutions:**
1. Reduce batch size: `--batch_size 8`
2. Reduce max length: `--max_length 256`
3. Enable gradient checkpointing (edit `train_model.py`, set `gradient_checkpointing=True`)
4. Clear GPU cache before training:
   ```python
   python -c "import torch; torch.cuda.empty_cache()"
   ```

### Issue: "CUDA not available"
**Solutions:**
1. Verify PyTorch installation:
   ```bash
   python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
   ```
2. Reinstall CUDA-enabled PyTorch (see Step 1)
3. Check NVIDIA driver version: `nvidia-smi`
4. Ensure CUDA Toolkit is installed

### Issue: GPU utilization is low (<50%)
**Solutions:**
1. Increase batch size
2. Enable data loader workers
3. Check if CPU is bottleneck (Task Manager)
4. Ensure data is preprocessed

### Issue: "torch.cuda.OutOfMemoryError" during evaluation
**Solution:**
Reduce evaluation batch size in `train_model.py`:
```python
per_device_eval_batch_size=8  # Reduce from 16
```

## 📊 Performance Tips

### 1. Maximize GPU Utilization
- Use largest batch size that fits in memory
- Enable mixed precision (BF16 recommended)
- Use data loader workers

### 2. Optimize Data Loading
- Pin memory to GPU
- Use SSD for data storage
- Preprocess data before training

### 3. Monitor Training
- Watch GPU utilization (should be >90%)
- Check TensorBoard for metrics
- Monitor memory usage

### 4. Batch Size Tuning
Start with recommended size and adjust:
```bash
# Test different batch sizes
python train_model.py --data_dir ./data --output_dir ./test_bs16 --bf16 --batch_size 16
python train_model.py --data_dir ./data --output_dir ./test_bs32 --bf16 --batch_size 32
python train_model.py --data_dir ./data --output_dir ./test_bs64 --bf16 --batch_size 64
```

## 📈 Expected Training Time

With GPU and BF16 on ~30,000 samples:

| Epochs | Batch Size | Estimated Time | GPU Utilization |
|--------|-----------|----------------|-----------------|
| 3      | 16        | ~1.5-2 hours   | ~80%            |
| 3      | 32        | ~1-1.5 hours   | ~95%            |
| 3      | 64        | ~0.75-1 hour   | ~98%            |
| 5      | 32        | ~1.5-2.5 hours | ~95%            |

**Without GPU (CPU only):**
- 3 epochs: ~12-18 hours ⚠️

## 🎓 Summary

### Best Configuration for Blackwell GPU:
```bash
python train_model.py \
  --data_dir ./data \
  --output_dir ./outputs \
  --bf16 \
  --batch_size 32 \
  --num_epochs 3 \
  --learning_rate 2e-5
```

### What This Enables:
- ⚡ **2-3x faster training** with BF16
- 🚀 **Higher batch size** for better convergence  
- 💾 **Lower memory usage** (50% reduction)
- 🎯 **Better stability** than FP16
- 📊 **Real-time progress tracking** with tqdm

## 🔗 Additional Resources

- [PyTorch CUDA Documentation](https://pytorch.org/docs/stable/cuda.html)
- [Mixed Precision Training Guide](https://pytorch.org/docs/stable/amp.html)
- [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/blackwell/)
- [Transformers Training Guide](https://huggingface.co/docs/transformers/perf_train_gpu_one)

---

**Ready to train?** Run `.\setup_gpu.ps1` to get started! 🚀
