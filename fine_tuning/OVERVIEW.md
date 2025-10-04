# Multi-Label Hate Speech Classification - Project Overview

## 📋 Project Summary

This project fine-tunes the **unbiased-toxic-roberta** model for multi-label hate speech classification using two combined datasets:
- `annotations_data_final.csv` (10,946 samples)
- `HatefulData_enriched_fixed.csv` (19,958 samples)

**Total dataset size**: ~30,904 samples

## 🎯 Classification Task

### Label Structure

The model performs **11-label classification** with a hierarchical constraint:

#### 1. Primary Label (Binary - Mutually Exclusive)
- `hate`: Text contains hateful content
- `nothate`: Text does not contain hateful content

#### 2. Secondary Labels (Multi-label - Only for "hate")
When text is classified as `hate`, it can have one or more of these 10 labels:

1. **Incitement of Violence** - Calls for violent action
2. **Praising violence** - Glorification of violent acts
3. **Praising extremist acts** - Support for extremism
4. **Targeting ethnic or racial groups** - Racial/ethnic discrimination
5. **Ideologically motivated threats** - Threats based on ideology
6. **Anti-democratic rhetoric** - Attacks on democratic values
7. **Personal Attacks** - Direct attacks on individuals
8. **Sexual harassment** - Sexual misconduct or harassment
9. **Physical violence** - References to physical harm
10. **Psychological attacks** - Psychological manipulation/harm

### Key Constraint
- **If `nothate`**: All 10 secondary labels MUST be 0
- **If `hate`**: Can have 0 to 10 secondary labels active

## 📁 Project Structure

```
fine_tuning/
├── prepare_data.py              # Data preprocessing and splitting
├── train_model.py               # Model training script
├── evaluate_model.py            # Comprehensive evaluation
├── inference.py                 # Prediction script for new text
├── config.json                  # Configuration file
├── requirements_finetuning.txt  # Python dependencies
├── run_pipeline.ps1             # Quick start script (PowerShell)
├── README.md                    # Detailed documentation
└── OVERVIEW.md                  # This file

data/                            # Created after prepare_data.py
├── train.csv                    # Training set (~70%)
├── val.csv                      # Validation set (~15%)
├── test.csv                     # Test set (~15%)
└── label_info.json             # Label configuration

outputs/                         # Created during training
├── best_model/                  # Best model checkpoint
│   ├── pytorch_model.bin
│   ├── config.json
│   └── tokenizer files
├── train_results.json           # Training metrics
├── test_results.json            # Test set metrics
└── logs/                        # TensorBoard logs

evaluation_results/              # Created during evaluation
├── evaluation_metrics.json      # Detailed metrics
├── confusion_matrices.json      # Confusion matrices
├── error_samples.json          # Sample errors for analysis
└── plots/                       # Visualization plots
    └── confusion_matrix_*.png
```

## 🚀 Quick Start

### Prerequisites
```bash
# Install dependencies
pip install -r requirements_finetuning.txt
```

### Option 1: Run Complete Pipeline (Automated)
```powershell
# Run from fine_tuning directory
.\run_pipeline.ps1
```

This will automatically:
1. Prepare and split the data
2. Train the model (3 epochs, ~2-4 hours on GPU)
3. Evaluate on test set
4. Generate comprehensive reports

### Option 2: Step-by-Step Execution

#### Step 1: Prepare Data
```bash
cd fine_tuning
python prepare_data.py
```

**Output**: 
- Combines and cleans datasets
- Validates label constraints
- Creates train/val/test splits
- Saves to `data/` directory

**Expected time**: 1-2 minutes

#### Step 2: Train Model
```bash
python train_model.py \
  --data_dir ./data \
  --output_dir ./outputs \
  --num_epochs 3 \
  --batch_size 16 \
  --learning_rate 2e-5
```

**Options**:
- Add `--fp16` for faster training with GPU
- Adjust `--batch_size` based on GPU memory
- Increase `--num_epochs` for potentially better results

**Expected time**:
- With GPU: 2-4 hours
- Without GPU: 8-12 hours

#### Step 3: Evaluate Model
```bash
python evaluate_model.py \
  --model_path ./outputs/best_model \
  --data_dir ./data \
  --output_dir ./evaluation_results
```

**Expected time**: 5-10 minutes

#### Step 4: Make Predictions
```bash
# Single text
python inference.py \
  --model_path ./outputs/best_model \
  --text "Your text here"

# Batch from file
python inference.py \
  --model_path ./outputs/best_model \
  --input_file texts.txt \
  --output_file predictions.json
```

## 📊 Expected Performance

Based on similar hate speech classification tasks:

### Primary Label (Hate/Nothate)
- **Accuracy**: 85-92%
- **F1 Score**: 0.80-0.90
- **Precision**: 0.82-0.92
- **Recall**: 0.78-0.88

### Secondary Labels (Multi-label)
- **F1 Score (Macro)**: 0.60-0.75
- **F1 Score (Micro)**: 0.65-0.80
- **Hamming Loss**: 0.05-0.15

Performance varies based on:
- Class distribution in dataset
- Model convergence during training
- Hyperparameter settings

## 🔧 Model Architecture

### Base Model
- **Name**: `unbiased-toxic-roberta`
- **Architecture**: RoBERTa-base (125M parameters)
- **Pre-training**: Fine-tuned on toxic language detection

### Custom Modifications
1. **Custom Classification Head**
   - Dropout layer (p=0.1)
   - Linear layer → 11 outputs

2. **Custom Loss Function**
   - Binary Cross-Entropy (BCE) for all labels
   - Constraint penalty term
   - Formula: `Loss = BCE + 0.5 * constraint_penalty`

3. **Constraint Enforcement**
   - During training: Penalty for activating secondary labels when primary=nothate
   - During inference: Force secondary labels to 0 when primary=nothate

## 📈 Monitoring Training

### TensorBoard
```bash
tensorboard --logdir ./outputs/logs
```

Then open http://localhost:6006 in your browser

### Key Metrics to Watch
- **Training Loss**: Should decrease steadily
- **Validation F1 (micro)**: Primary metric for model selection
- **Primary Accuracy**: Should be >85%
- **Secondary F1**: Should improve over epochs

## 🎛️ Hyperparameter Tuning

### Critical Parameters

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `learning_rate` | 2e-5 | 1e-5 to 5e-5 | Higher = faster learning but less stable |
| `num_epochs` | 3 | 3-10 | More epochs may improve performance |
| `batch_size` | 16 | 8-32 | Larger = faster but needs more memory |
| `warmup_ratio` | 0.1 | 0.0-0.2 | Warmup helps stabilize training |
| `weight_decay` | 0.01 | 0.0-0.1 | Regularization to prevent overfitting |

### Tuning Strategy
1. Start with defaults
2. If underfitting: increase epochs, reduce weight_decay
3. If overfitting: reduce epochs, increase weight_decay, add more data
4. If unstable: reduce learning_rate, increase warmup_ratio

## 🐛 Troubleshooting

### CUDA Out of Memory
**Solutions**:
- Reduce `batch_size` (try 8 or 4)
- Reduce `max_length` (try 256)
- Use gradient accumulation
- Close other GPU-using applications

### Poor Primary Label Performance
**Solutions**:
- Check data quality and class balance
- Increase training epochs
- Try different learning rates
- Check for data leakage

### Poor Secondary Label Performance
**Solutions**:
- Check class imbalance (some labels may be very rare)
- Adjust constraint penalty weight in loss function
- Use class weights in loss calculation
- Consider focal loss for imbalanced labels

### Model Not Loading
**Solutions**:
- Verify all dependencies are installed
- Check model path is correct
- Ensure training completed successfully
- Check disk space for saved models

## 📝 Usage Examples

### Example 1: Basic Training
```bash
python train_model.py \
  --data_dir ./data \
  --output_dir ./outputs
```

### Example 2: GPU Training with Mixed Precision
```bash
python train_model.py \
  --data_dir ./data \
  --output_dir ./outputs \
  --fp16 \
  --batch_size 32
```

### Example 3: Extended Training
```bash
python train_model.py \
  --data_dir ./data \
  --output_dir ./outputs \
  --num_epochs 10 \
  --learning_rate 1e-5
```

### Example 4: Batch Prediction
```python
from inference import HateSpeechPredictor

predictor = HateSpeechPredictor("./outputs/best_model")

texts = [
    "This is a sample text",
    "Another example text",
]

results = predictor.predict_batch(texts, return_probs=True)

for result in results:
    print(f"Text: {result['text']}")
    print(f"Classification: {result['primary_classification']}")
    print(f"Secondary labels: {result['secondary_labels']}")
    print()
```

## 🔬 Advanced Features

### Custom Configuration
Edit `config.json` to customize all parameters without command-line arguments.

### Model Checkpointing
- Best model saved based on validation F1 score
- Can resume training from checkpoints
- Saves top 2 checkpoints to save disk space

### Comprehensive Evaluation
- Per-label metrics
- Confusion matrices for all labels
- Error analysis with sample errors
- Visual plots for interpretation

## 📚 Additional Resources

### Documentation
- **README.md**: Detailed setup and usage guide
- **config.json**: All configurable parameters
- **Code comments**: Extensive inline documentation

### Model Citation
```bibtex
@misc{hanu2020unbiased,
    title={Unbiased Toxic Language Detection},
    author={Hanu, Laura and Unitary team},
    year={2020}
}
```

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review error messages carefully
3. Verify all dependencies are installed
4. Check that data files are in correct locations

## ✅ Checklist

Before training:
- [ ] Datasets in correct location (`../data/`)
- [ ] Dependencies installed (`pip install -r requirements_finetuning.txt`)
- [ ] Sufficient disk space (>5GB for models)
- [ ] GPU available (optional but recommended)

After training:
- [ ] Model saved in `outputs/best_model/`
- [ ] Training metrics look reasonable
- [ ] Evaluation completed successfully
- [ ] Test predictions working correctly

## 🎓 Learning Resources

To understand the model better:
- RoBERTa paper: [Liu et al., 2019](https://arxiv.org/abs/1907.11692)
- Multi-label classification: [Read et al., 2011](https://link.springer.com/article/10.1007/s10994-011-5256-5)
- Hate speech detection: Various recent papers on arXiv

---

**Good luck with your fine-tuning!** 🚀
