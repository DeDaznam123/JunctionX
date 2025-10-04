# Hate Speech Classification with unbiased-toxic-roberta

This project fine-tunes the `unitary/unbiased-toxic-roberta` model on your HatefulData dataset to classify text as hate speech or not.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Your Data

Make sure your `HatefulData.csv` is in the correct location:
- Path: `c:\Users\mitev\DELFT\JunctionX\HatefulData.csv`
- Required columns: `text`, `label` (with values 'hate' or 'nothate')

## Usage

### Training the Model

Run the fine-tuning script:

```bash
python hate_speech_finetuning.py
```

**What happens during training:**
1. Loads the HatefulData.csv dataset
2. Splits data into train (70%), validation (15%), and test (15%) sets
3. Fine-tunes the unbiased-toxic-roberta model
4. Evaluates the model and saves results
5. Saves the trained model to `models/hate_speech_model/`

**Training parameters:**
- Epochs: 3
- Batch size: 16
- Learning rate: 2e-5
- Max sequence length: 128 tokens
- Early stopping with patience: 2

**Output files:**
- `models/hate_speech_model/` - Fine-tuned model and tokenizer
- `models/hate_speech_model/test_predictions.csv` - Predictions on test set
- `models/hate_speech_model/logs/` - Training logs

### Using the Trained Model for Inference

After training, use the inference script to classify new texts:

```bash
python inference.py
```

**Or use it in your code:**

```python
from inference import HateSpeechClassifier

# Load the trained model
classifier = HateSpeechClassifier(r"c:\Users\mitev\DELFT\JunctionX\models\hate_speech_model")

# Classify a single text
text = "Your text here"
prediction, probabilities = classifier.predict(text, return_probabilities=True)

print(f"Prediction: {prediction}")
print(f"Probabilities: {probabilities}")

# Classify multiple texts
texts = ["text1", "text2", "text3"]
predictions, probabilities = classifier.predict_batch(texts)
```

## Model Information

**Base Model:** `unitary/unbiased-toxic-roberta`
- A RoBERTa model specifically designed to be less biased in toxic content detection
- Pre-trained on a large corpus of toxic and non-toxic text
- Fine-tuned on your specific hate speech dataset

**Task:** Binary classification
- Class 0: nothate
- Class 1: hate

## Expected Results

The model should achieve:
- Accuracy: ~85-95% (depending on your dataset)
- F1-score: High F1 for both classes
- Low bias towards specific groups

## Troubleshooting

### GPU Memory Issues
If you encounter GPU memory errors, reduce the batch size in `hate_speech_finetuning.py`:
```python
BATCH_SIZE = 8  # or 4
```

### CPU Training
The script automatically detects if CUDA is available. If not, it will train on CPU (slower but works).

### Dataset Issues
Make sure your CSV has the correct format:
- Column `text`: The text to classify
- Column `label`: Must be either 'hate' or 'nothate'

## Customization

You can modify these parameters in `hate_speech_finetuning.py`:

```python
MAX_LENGTH = 128        # Maximum sequence length
BATCH_SIZE = 16         # Batch size for training
EPOCHS = 3              # Number of training epochs
LEARNING_RATE = 2e-5    # Learning rate
```

## Files Description

- `hate_speech_finetuning.py` - Main training script
- `inference.py` - Inference script for using the trained model
- `requirements.txt` - Python dependencies
- `README_FINETUNING.md` - This file

## Next Steps

After training:
1. Check the test results in the console output
2. Review `test_predictions.csv` for detailed predictions
3. Use `inference.py` to classify new texts
4. Integrate the classifier into your application

## Performance Monitoring

The training script outputs:
- Loss values during training
- Validation metrics after each epoch
- Final test set evaluation with precision, recall, F1-score
- Classification report with per-class metrics

Monitor these to ensure the model is learning effectively!
