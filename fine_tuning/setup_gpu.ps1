# GPU Setup Script for NVIDIA Blackwell
# This script installs CUDA-enabled PyTorch and configures the environment

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "NVIDIA Blackwell GPU Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$PYTHON = "C:/Users/mitev/AppData/Local/Programs/Python/Python313/python.exe"

# Step 1: Check current PyTorch installation
Write-Host "Checking current PyTorch installation..." -ForegroundColor Yellow
& $PYTHON -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
Write-Host ""

# Step 2: Uninstall CPU-only PyTorch
Write-Host "Uninstalling CPU-only PyTorch versions..." -ForegroundColor Yellow
& $PYTHON -m pip uninstall -y torch torchvision torchaudio
Write-Host ""

# Step 3: Install CUDA-enabled PyTorch for NVIDIA Blackwell (requires CUDA 12.x)
Write-Host "Installing CUDA-enabled PyTorch (CUDA 12.4)..." -ForegroundColor Yellow
Write-Host "This optimized for NVIDIA Blackwell architecture..." -ForegroundColor Gray
& $PYTHON -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
Write-Host ""

# Step 4: Verify installation
Write-Host "Verifying GPU setup..." -ForegroundColor Yellow
& $PYTHON -c @"
import torch
print('='*60)
print('PyTorch GPU Configuration')
print('='*60)
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Number of GPUs: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        props = torch.cuda.get_device_properties(i)
        print(f'  - Compute Capability: {props.major}.{props.minor}')
        print(f'  - Total Memory: {props.total_memory / 1024**3:.2f} GB')
        print(f'  - Multi-Processors: {props.multi_processor_count}')
    print(f'Current device: {torch.cuda.current_device()}')
    print(f'Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}')
    
    # Test tensor operation on GPU
    print('\nTesting GPU operations...')
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print('✓ GPU computation successful!')
else:
    print('✗ CUDA not available - GPU will not be used')
print('='*60)
"@
Write-Host ""

# Step 5: Install additional GPU acceleration libraries
Write-Host "Installing additional GPU optimization libraries..." -ForegroundColor Yellow
& $PYTHON -m pip install ninja  # For faster compilation
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "GPU Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Run training with --fp16 flag for mixed precision (faster on Blackwell)" -ForegroundColor White
Write-Host "2. Consider using larger batch sizes to maximize GPU utilization" -ForegroundColor White
Write-Host "3. Use --bf16 flag if supported (better for Blackwell architecture)" -ForegroundColor White
Write-Host ""
Write-Host "Example command:" -ForegroundColor Yellow
Write-Host "  python train_model.py --data_dir ./data --output_dir ./outputs --fp16 --batch_size 32" -ForegroundColor White
Write-Host ""
