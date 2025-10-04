# Quick Start Script for Fine-tuning unbiased-toxic-roberta
# This script runs the complete pipeline: data preparation, training, and evaluation

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Fine-tuning unbiased-toxic-roberta" -ForegroundColor Cyan
Write-Host "Multi-label Hate Speech Classification" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Set Python executable
$PYTHON = "C:/Users/mitev/AppData/Local/Programs/Python/Python313/python.exe"

# Step 1: Data Preparation
Write-Host "Step 1: Preparing data..." -ForegroundColor Yellow
& $PYTHON prepare_data.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Data preparation failed!" -ForegroundColor Red
    exit 1
}
Write-Host "Data preparation completed!" -ForegroundColor Green
Write-Host ""

# Step 2: Training
Write-Host "Step 2: Training model..." -ForegroundColor Yellow
Write-Host "This may take a while depending on your hardware..." -ForegroundColor Gray
& $PYTHON train_model.py --data_dir ./data --output_dir ./outputs --num_epochs 3 --batch_size 16 --learning_rate 2e-5
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Training failed!" -ForegroundColor Red
    exit 1
}
Write-Host "Training completed!" -ForegroundColor Green
Write-Host ""

# Step 3: Evaluation
Write-Host "Step 3: Evaluating model..." -ForegroundColor Yellow
& $PYTHON evaluate_model.py --model_path ./outputs/best_model --data_dir ./data --output_dir ./evaluation_results
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Evaluation failed!" -ForegroundColor Red
    exit 1
}
Write-Host "Evaluation completed!" -ForegroundColor Green
Write-Host ""

# Complete
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Pipeline completed successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Results:" -ForegroundColor Yellow
Write-Host "  - Trained model: ./outputs/best_model/" -ForegroundColor White
Write-Host "  - Training metrics: ./outputs/train_results.json" -ForegroundColor White
Write-Host "  - Test metrics: ./outputs/test_results.json" -ForegroundColor White
Write-Host "  - Evaluation results: ./evaluation_results/" -ForegroundColor White
Write-Host ""
Write-Host "To make predictions on new text:" -ForegroundColor Yellow
Write-Host "  $PYTHON inference.py --model_path ./outputs/best_model --text ""Your text here""" -ForegroundColor White
Write-Host ""
