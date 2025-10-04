# 🎯 Quick Reference - GPU Training Commands

## 🚀 Setup (One-time)

```powershell
cd fine_tuning
.\setup_gpu.ps1
```

## ⚡ Training Commands

### Best for NVIDIA Blackwell GPU
```bash
python train_model.py --data_dir ./data --output_dir ./outputs --bf16 --batch_size 32
```

### If BF16 not supported
```bash
python train_model.py --data_dir ./data --output_dir ./outputs --fp16 --batch_size 32
```

### CPU Only (slow)
```bash
python train_model.py --data_dir ./data --output_dir ./outputs --batch_size 16
```

## 📊 Monitoring

### GPU Usage
```bash
nvidia-smi -l 1
```

### Training Progress
```bash
tensorboard --logdir ./outputs/logs
```

## 🔧 Flags

| Flag | Description | When to Use |
|------|-------------|-------------|
| `--bf16` | BFloat16 precision | Blackwell, Hopper, Ampere GPUs |
| `--fp16` | Float16 precision | Older NVIDIA GPUs |
| `--batch_size 32` | Larger batches | With GPU, more VRAM |
| `--batch_size 16` | Default batches | CPU or limited VRAM |
| `--num_epochs 5` | More training | Better accuracy |

## ⚠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `--batch_size` to 8 or 16 |
| CUDA not available | Run `.\setup_gpu.ps1` |
| Slow training | Add `--bf16` or `--fp16` |
| GPU not used | Check with `nvidia-smi` |

## 📈 Expected Speed

| Configuration | Time for 3 epochs |
|--------------|-------------------|
| CPU Only | ~12-18 hours |
| GPU FP32 | ~2-3 hours |
| GPU FP16 | ~1-1.5 hours |
| GPU BF16 | ~1-1.5 hours |

---

**Full Guide**: See [GPU_TRAINING_GUIDE.md](GPU_TRAINING_GUIDE.md)
