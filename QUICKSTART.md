# 🚀 QUICK START GUIDE - Hate Speech Classification

## ✅ Setup Complete!

Your hate speech classification project is ready to go! Here's what has been set up:

### 📁 Files Created:
1. **`hate_speech_finetuning.py`** - Main training script
2. **`inference.py`** - Script to use the trained model
3. **`requirements.txt`** - Python dependencies
4. **`run.ps1`** - Interactive menu for easy execution
5. **`README_FINETUNING.md`** - Detailed documentation

### 📦 Dependencies Installed:
- ✓ PyTorch
- ✓ Transformers (Hugging Face)
- ✓ Datasets
- ✓ Pandas, NumPy
- ✓ Scikit-learn
- ✓ Accelerate

---

## 🎯 How to Use

### Option 1: Interactive Menu (Easiest)
```powershell
.\run.ps1
```
This will show you an interactive menu with options to train, test, and use your model.

### Option 2: Direct Training
```powershell
python hate_speech_finetuning.py
```

### Option 3: Direct Inference (after training)
```powershell
python inference.py
```

---

## 📊 What the Training Does

The `hate_speech_finetuning.py` script will:

1. **Load your data** from `HatefulData.csv` (40,754 examples)
2. **Split the data:**
   - 70% for training
   - 15% for validation
   - 15% for testing
3. **Fine-tune** the unbiased-toxic-roberta model
4. **Evaluate** and report metrics
5. **Save** the trained model to `models/hate_speech_model/`

### Expected Output:
- Training takes ~30-60 minutes (depending on hardware)
- Final model with ~85-95% accuracy
- Detailed metrics (precision, recall, F1-score)
- Test predictions saved to CSV

---

## 💡 Using the Trained Model

After training, you can classify any text:

```python
from inference import HateSpeechClassifier

# Load the model
classifier = HateSpeechClassifier(
    r"c:\Users\mitev\DELFT\JunctionX\models\hate_speech_model"
)

# Classify a text
text = "Your text here"
prediction, probs = classifier.predict(text, return_probabilities=True)

print(f"Prediction: {prediction}")
# Output: 'hate' or 'nothate'

print(f"Confidence: {probs}")
# Output: {'nothate': 0.95, 'hate': 0.05}
```

---

## 🔧 Configuration

Current settings (can be modified in `hate_speech_finetuning.py`):

```python
MODEL_NAME = "unitary/unbiased-toxic-roberta"
MAX_LENGTH = 128        # Token length
BATCH_SIZE = 16         # Adjust if GPU memory issues
EPOCHS = 3              # Training epochs
LEARNING_RATE = 2e-5    # Learning rate
```

### For GPU Memory Issues:
Reduce `BATCH_SIZE` to 8 or 4

### For Better Accuracy:
Increase `EPOCHS` to 5 or more (with early stopping)

---

## 📈 Model Performance

The unbiased-toxic-roberta model is:
- **Pre-trained** on toxic content detection
- **Designed** to reduce bias in predictions
- **Optimized** for hate speech classification

Expected metrics on your dataset:
- Accuracy: 85-95%
- F1-Score: 0.85-0.95
- Low false positive rate

---

## 🎓 Next Steps

1. **Train the model:**
   ```powershell
   python hate_speech_finetuning.py
   ```

2. **Check the results:**
   - Look at console output for metrics
   - Review `models/hate_speech_model/test_predictions.csv`

3. **Use it for inference:**
   ```powershell
   python inference.py
   ```

4. **Integrate into your app:**
   ```python
   from inference import HateSpeechClassifier
   classifier = HateSpeechClassifier("models/hate_speech_model")
   result = classifier.predict("your text")
   ```

---

## 🆘 Troubleshooting

### "Model not found" error
- You need to train the model first
- Run: `python hate_speech_finetuning.py`

### GPU/CUDA errors
- The script works on CPU too (just slower)
- Or reduce BATCH_SIZE in the script

### Import errors
- Reinstall dependencies: `pip install -r requirements.txt`

### Dataset errors
- Make sure `HatefulData.csv` is in the correct location
- Check it has 'text' and 'label' columns

---

## 📚 Additional Resources

- **Full Documentation:** See `README_FINETUNING.md`
- **Model Card:** https://huggingface.co/unitary/unbiased-toxic-roberta
- **Transformers Docs:** https://huggingface.co/docs/transformers

---

## ✨ Ready to Go!

You're all set! Just run:

```powershell
python hate_speech_finetuning.py
```

And watch your model train! 🚀

---

**Questions?** Check `README_FINETUNING.md` for detailed information.
