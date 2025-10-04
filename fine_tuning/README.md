# Fine-tuning unbiased-toxic-roberta for Multi-label Hate Speech Classification

This directory contains scripts for fine-tuning the `unbiased-toxic-roberta` model on a custom hate speech dataset with multi-label classification.

## Overview

The model performs **11-label classification**:
1. **Primary Label (Binary)**: `hate` or `nothate` (mutually exclusive)
2. **Secondary Labels (Multi-label, 10 categories)**: Only applicable when primary label is `hate`
   - Incitement of Violence
   - Praising violence
   - Praising extremist acts
   - Targeting ethnic or racial groups
   - Ideologically motivated threats
   - Anti-democratic rhetoric
   - Personal Attacks
   - Sexual harassment
   - Physical violence
   - Psychological attacks

### Key Constraint
- If the text is classified as `nothate`, it **cannot** have any of the 10 secondary labels
- If the text is classified as `hate`, it **can** have one or more secondary labels

## Files

- `prepare_data.py`: Data preprocessing and preparation script
- `train_model.py`: Model training script with custom architecture
- `evaluate_model.py`: Comprehensive evaluation script
- `config.json`: Configuration file for hyperparameters
- `README.md`: This file

## Requirements

Install the required packages:

```bash
pip install transformers datasets torch scikit-learn accelerate evaluate tensorboard pandas numpy matplotlib seaborn
```

Or use the requirements file from the project root.

## Usage

### Step 1: Prepare the Data

First, prepare and preprocess the datasets:

```bash
cd fine_tuning
python prepare_data.py
```

This will:
- Load `annotations_data_final.csv` and `HatefulData_enriched_fixed.csv`
- Combine and clean the data
- Validate label constraints
- Split into train/validation/test sets (70%/15%/15%)
- Save processed data to `./data/` directory

Output:
- `data/train.csv`
- `data/val.csv`
- `data/test.csv`
- `data/label_info.json`

### Step 2: Setup GPU (Recommended)

**For NVIDIA GPU users** (including Blackwell architecture):

Run the GPU setup script to install CUDA-enabled PyTorch:
```bash
.\setup_gpu.ps1
```

This ensures your GPU is properly configured for fast training. See [GPU_TRAINING_GUIDE.md](GPU_TRAINING_GUIDE.md) for detailed instructions.

### Step 3: Train the Model

**With GPU (Recommended - 10-20x faster):**

```bash
# For NVIDIA Blackwell or modern GPUs (best performance)
python train_model.py \
  --data_dir ./data \
  --output_dir ./outputs \
  --bf16 \
  --batch_size 32

# For older GPUs or if bf16 not supported
python train_model.py \
  --data_dir ./data \
  --output_dir ./outputs \
  --fp16 \
  --batch_size 32
```

**Without GPU (slower):**

```bash
python train_model.py --data_dir ./data --output_dir ./outputs
```

**Advanced options:**

```bash
python train_model.py \
  --model_name unbiased-toxic-roberta \
  --data_dir ./data \
  --output_dir ./outputs \
  --num_epochs 5 \
  --batch_size 32 \
  --learning_rate 2e-5 \
  --max_length 512 \
  --bf16
```

**Mixed Precision Options:**
- `--bf16`: BFloat16 precision (recommended for NVIDIA Blackwell, Hopper, Ampere GPUs)
- `--fp16`: Float16 precision (good for older GPUs)
- No flag: Full FP32 precision (slowest, most precise)

Training outputs:
- `outputs/best_model/`: Best model checkpoint
- `outputs/train_results.json`: Training metrics
- `outputs/test_results.json`: Test set evaluation metrics
- `outputs/logs/`: TensorBoard logs

### Step 4: Monitor Training (Optional)

Monitor training progress with TensorBoard:

```bash
tensorboard --logdir ./outputs/logs
```

Monitor GPU usage (if using GPU):
```bash
nvidia-smi -l 1
```

### Step 5: Evaluate the Model

Run comprehensive evaluation on the test set:

```bash
python evaluate_model.py \
  --model_path ./outputs/best_model \
  --data_dir ./data \
  --output_dir ./evaluation_results
```

Evaluation outputs:
- `evaluation_results/evaluation_metrics.json`: Overall and per-label metrics
- `evaluation_results/confusion_matrices.json`: Confusion matrices for all labels
- `evaluation_results/error_samples.json`: Sample of prediction errors
- `evaluation_results/plots/`: Confusion matrix visualizations

## Model Architecture

The model uses a custom architecture:

1. **Base**: `unbiased-toxic-roberta` (RoBERTa-base fine-tuned on toxic language)
2. **Custom Head**: Linear layer with 11 outputs
3. **Loss Function**: 
   - Binary Cross-Entropy with Logits Loss (BCE)
   - Additional constraint penalty to enforce label dependencies

### Custom Loss Function

The loss function enforces the constraint that secondary labels should be inactive when the primary label is `nothate`:

```python
total_loss = BCE_loss + 0.5 * constraint_penalty
```

Where `constraint_penalty` penalizes secondary label activation for `nothate` samples.

## Evaluation Metrics

The evaluation script computes:

### Overall Metrics
- Accuracy
- F1 Score (micro, macro, weighted)
- Precision & Recall (micro, macro)
- Hamming Loss

### Primary Label Metrics
- Accuracy, F1, Precision, Recall for hate/nothate classification

### Secondary Label Metrics
- Per-label F1, Precision, Recall (for the 10 secondary categories)
- Metrics computed only on hate samples

## Expected Performance

Based on similar hate speech classification tasks:
- **Primary Label Accuracy**: 85-92%
- **Primary Label F1**: 0.80-0.90
- **Secondary Labels F1 (macro)**: 0.60-0.75

Performance may vary based on:
- Training data quality and size
- Class imbalance
- Hyperparameter tuning

## Hyperparameter Tuning

Key hyperparameters to tune (in `config.json` or via command-line args):

- `learning_rate`: 1e-5 to 5e-5 (default: 2e-5)
- `num_epochs`: 3-10 (default: 3)
- `batch_size`: 8-32 (default: 16)
- `warmup_ratio`: 0.0-0.2 (default: 0.1)
- `weight_decay`: 0.0-0.1 (default: 0.01)

## Advanced Usage

### Using Custom Configuration

Edit `config.json` and load it in your script:

```python
import json

with open('config.json', 'r') as f:
    config = json.load(f)

trainer = HateSpeechTrainer(
    model_name=config['model']['name'],
    data_dir=config['paths']['data_dir'],
    output_dir=config['paths']['output_dir'],
)

trainer.train(
    num_epochs=config['training']['num_epochs'],
    batch_size=config['training']['batch_size'],
    learning_rate=config['training']['learning_rate'],
)
```

### Inference on New Text

```python
from transformers import AutoTokenizer
import torch

# Load model and tokenizer
model_path = "./outputs/best_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = MultiLabelRobertaWithConstraints(model_path, num_labels=11)
model.eval()

# Prepare text
text = "Your text here"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

# Get predictions
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs['logits']
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).int()

# Enforce constraint
if preds[0, 0] == 0:  # if nothate
    preds[0, 1:] = 0

print("Predictions:", preds)
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size`
- Reduce `max_length`
- Use gradient accumulation: increase `gradient_accumulation_steps`

### Poor Performance on Secondary Labels
- Check class imbalance (use weighted loss)
- Increase training epochs
- Adjust constraint penalty weight in loss function

### Model Not Loading
- Ensure all dependencies are installed
- Check model path is correct
- Verify model files exist in output directory

## Citation

If using the unbiased-toxic-roberta model:
```
@misc{hanu2020unbiased,
    title={Unbiased Toxic Language Detection},
    author={Hanu, Laura and Unitary team},
    year={2020}
}
```

## License

This fine-tuning code is provided as-is for educational and research purposes.
