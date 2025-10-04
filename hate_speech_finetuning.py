import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN messages
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # Better memory management
os.environ['TORCH_CUDNN_SDPA_ENABLED'] = '0'  # Disable scaled dot product attention
warnings.filterwarnings('ignore', category=UserWarning)  # Ignore user warnings
warnings.filterwarnings('ignore', category=FutureWarning)  # Ignore future warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset as HFDataset
from tqdm.auto import tqdm

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

class HateSpeechDataset(Dataset):
    """Custom Dataset for hate speech classification"""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_and_prepare_data(csv_path):
    """Load and prepare the HatefulData dataset"""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Display dataset info
    print(f"\nDataset shape: {df.shape}")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())
    
    # Convert labels to binary (0: nothate, 1: hate)
    df['binary_label'] = (df['label'] == 'hate').astype(int)
    
    # Get texts and labels
    texts = df['text'].tolist()
    labels = df['binary_label'].tolist()
    
    return texts, labels, df

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    acc = accuracy_score(labels, predictions)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    # Configuration
    MODEL_NAME = "unitary/unbiased-toxic-roberta"
    DATA_PATH = r"c:\Users\mitev\DELFT\JunctionX\HatefulData.csv"
    OUTPUT_DIR = r"c:\Users\mitev\DELFT\JunctionX\models\hate_speech_model"
    MAX_LENGTH = 128
    BATCH_SIZE = 32  # Increased for faster training
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    USE_SAMPLE = False  # Set to True to train on a sample for faster testing
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data
    texts, labels, df = load_and_prepare_data(DATA_PATH)
    
    # Optional: Use sample for faster testing
    if USE_SAMPLE:
        print("\n⚠️  Using sample data for faster training (5000 examples)")
        sample_size = min(5000, len(texts))
        indices = np.random.choice(len(texts), sample_size, replace=False)
        texts = [texts[i] for i in indices]
        labels = [labels[i] for i in indices]
    
    # Split data into train, validation, and test sets
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.3, random_state=RANDOM_SEED, stratify=labels
    )
    
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=RANDOM_SEED, stratify=temp_labels
    )
    
    print(f"\nTrain size: {len(train_texts)}")
    print(f"Validation size: {len(val_texts)}")
    print(f"Test size: {len(test_texts)}")
    
    # Load tokenizer and model directly from HuggingFace (faster!)
    print(f"\n📥 Loading tokenizer and model from HuggingFace: {MODEL_NAME}")
    print("This will be faster as it loads directly from the hub...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True  # Use fast tokenizer for speed
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "nothate", 1: "hate"},
        label2id={"nothate": 0, "hate": 1},
        ignore_mismatched_sizes=True,  # Ignore size mismatch in classification head
        torch_dtype=torch.float32,  # Use FP32 for compatibility with newer GPUs
        attn_implementation="eager",  # Use standard attention instead of flash attention
        problem_type="single_label_classification"  # Specify classification problem type
    )
    
    print("✅ Model loaded successfully!")
    
    # Prepare datasets for Hugging Face Trainer
    train_dataset = HFDataset.from_dict({
        'text': train_texts,
        'label': train_labels
    })
    
    val_dataset = HFDataset.from_dict({
        'text': val_texts,
        'label': val_labels
    })
    
    test_dataset = HFDataset.from_dict({
        'text': test_texts,
        'label': test_labels
    })
    
    # Tokenize datasets with progress bar
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH
        )
    
    print("\n🔄 Tokenizing datasets...")
    train_dataset = train_dataset.map(
        tokenize_function, 
        batched=True,
        desc="Tokenizing train set"
    )
    val_dataset = val_dataset.map(
        tokenize_function, 
        batched=True,
        desc="Tokenizing validation set"
    )
    test_dataset = test_dataset.map(
        tokenize_function, 
        batched=True,
        desc="Tokenizing test set"
    )
    print("✅ Tokenization complete!")
    
    # Set format for PyTorch
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    
    # Define training arguments (optimized for speed)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,  # Larger batch for eval (faster)
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        eval_strategy="epoch",  # Updated parameter name
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=f'{OUTPUT_DIR}/logs',
        logging_steps=50,  # More frequent logging
        warmup_steps=500,
        save_total_limit=2,
        fp16=False,  # Disabled for compatibility with newer GPU architectures
        dataloader_num_workers=0,  # Set to 0 for Windows compatibility
        dataloader_pin_memory=True,  # Faster data transfer to GPU
        gradient_accumulation_steps=1,
        report_to="none",  # Disable wandb/tensorboard
        disable_tqdm=False,  # Keep progress bars enabled
        logging_first_step=True,
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train the model
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")
    
    trainer.train()
    
    # Evaluate on test set
    print("\n" + "="*50)
    print("Evaluating on test set...")
    print("="*50 + "\n")
    
    test_results = trainer.evaluate(test_dataset)
    print("\nTest Results:")
    for key, value in test_results.items():
        print(f"{key}: {value:.4f}")
    
    # Save the model and tokenizer
    print(f"\nSaving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Generate detailed predictions on test set
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    
    print("\n" + "="*50)
    print("Detailed Classification Report:")
    print("="*50 + "\n")
    print(classification_report(
        test_labels,
        pred_labels,
        target_names=['nothate', 'hate']
    ))
    
    # Save test predictions
    test_df = pd.DataFrame({
        'text': test_texts,
        'true_label': test_labels,
        'predicted_label': pred_labels,
        'correct': [t == p for t, p in zip(test_labels, pred_labels)]
    })
    test_df.to_csv(f'{OUTPUT_DIR}/test_predictions.csv', index=False)
    print(f"\nTest predictions saved to {OUTPUT_DIR}/test_predictions.csv")
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print("="*50)

if __name__ == "__main__":
    main()
