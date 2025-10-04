# Quick Start Script for Hate Speech Classification

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "Hate Speech Classification - Quick Start" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""

$pythonExe = "C:/Users/mitev/AppData/Local/Programs/Python/Python313/python.exe"
$scriptDir = "c:\Users\mitev\DELFT\JunctionX\JunctionX"

function Show-Menu {
    Write-Host "What would you like to do?" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "1. Train the model (fine-tune unbiased-toxic-roberta)" -ForegroundColor Green
    Write-Host "2. Run inference (classify texts with trained model)" -ForegroundColor Green
    Write-Host "3. Install/Update dependencies" -ForegroundColor Green
    Write-Host "4. View model information" -ForegroundColor Green
    Write-Host "5. Exit" -ForegroundColor Red
    Write-Host ""
}

function Train-Model {
    Write-Host ""
    Write-Host "=" * 70 -ForegroundColor Cyan
    Write-Host "Starting Model Training..." -ForegroundColor Cyan
    Write-Host "=" * 70 -ForegroundColor Cyan
    Write-Host ""
    Write-Host "This will:" -ForegroundColor Yellow
    Write-Host "- Load your HatefulData.csv" -ForegroundColor White
    Write-Host "- Split into train/validation/test sets" -ForegroundColor White
    Write-Host "- Fine-tune unbiased-toxic-roberta model" -ForegroundColor White
    Write-Host "- Save the trained model" -ForegroundColor White
    Write-Host ""
    Write-Host "Note: This may take 30-60 minutes depending on your hardware" -ForegroundColor Yellow
    Write-Host ""
    
    $confirm = Read-Host "Continue? (y/n)"
    if ($confirm -eq "y" -or $confirm -eq "Y") {
        Set-Location $scriptDir
        & $pythonExe hate_speech_finetuning.py
    }
}

function Run-Inference {
    Write-Host ""
    Write-Host "=" * 70 -ForegroundColor Cyan
    Write-Host "Running Inference..." -ForegroundColor Cyan
    Write-Host "=" * 70 -ForegroundColor Cyan
    Write-Host ""
    
    $modelPath = "c:\Users\mitev\DELFT\JunctionX\models\hate_speech_model"
    
    if (-Not (Test-Path $modelPath)) {
        Write-Host "Error: Trained model not found!" -ForegroundColor Red
        Write-Host "Please train the model first (option 1)" -ForegroundColor Yellow
        Write-Host ""
        return
    }
    
    Set-Location $scriptDir
    & $pythonExe inference.py
}

function Install-Dependencies {
    Write-Host ""
    Write-Host "=" * 70 -ForegroundColor Cyan
    Write-Host "Installing Dependencies..." -ForegroundColor Cyan
    Write-Host "=" * 70 -ForegroundColor Cyan
    Write-Host ""
    
    Set-Location $scriptDir
    & $pythonExe -m pip install -r requirements.txt
    
    Write-Host ""
    Write-Host "Dependencies installed successfully!" -ForegroundColor Green
}

function Show-Info {
    Write-Host ""
    Write-Host "=" * 70 -ForegroundColor Cyan
    Write-Host "Model Information" -ForegroundColor Cyan
    Write-Host "=" * 70 -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Base Model: unitary/unbiased-toxic-roberta" -ForegroundColor White
    Write-Host "Task: Binary Classification (hate / nothate)" -ForegroundColor White
    Write-Host "Dataset: HatefulData.csv" -ForegroundColor White
    Write-Host ""
    Write-Host "Training Configuration:" -ForegroundColor Yellow
    Write-Host "  - Epochs: 3" -ForegroundColor White
    Write-Host "  - Batch Size: 16" -ForegroundColor White
    Write-Host "  - Learning Rate: 2e-5" -ForegroundColor White
    Write-Host "  - Max Sequence Length: 128 tokens" -ForegroundColor White
    Write-Host ""
    Write-Host "Output Directory: c:\Users\mitev\DELFT\JunctionX\models\hate_speech_model" -ForegroundColor White
    Write-Host ""
    
    $modelPath = "c:\Users\mitev\DELFT\JunctionX\models\hate_speech_model"
    if (Test-Path $modelPath) {
        Write-Host "Model Status: TRAINED ✓" -ForegroundColor Green
    } else {
        Write-Host "Model Status: NOT TRAINED ✗" -ForegroundColor Red
    }
    Write-Host ""
}

# Main loop
while ($true) {
    Show-Menu
    $choice = Read-Host "Enter your choice (1-5)"
    
    switch ($choice) {
        "1" { Train-Model }
        "2" { Run-Inference }
        "3" { Install-Dependencies }
        "4" { Show-Info }
        "5" { 
            Write-Host ""
            Write-Host "Goodbye!" -ForegroundColor Cyan
            Write-Host ""
            exit 
        }
        default { 
            Write-Host ""
            Write-Host "Invalid choice. Please enter 1-5." -ForegroundColor Red
            Write-Host ""
        }
    }
    
    Write-Host ""
    Write-Host "Press any key to continue..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    Clear-Host
}
