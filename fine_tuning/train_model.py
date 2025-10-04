"""
Fine-tuning script for unbiased-toxic-roberta model on multi-label hate speech classification.

Model architecture:
- Base: unbiased-toxic-roberta
- Output: 11 labels (1 primary binary + 10 secondary multi-label)
- Loss: Custom loss combining BCE for primary and secondary labels with constraint enforcement
"""

import os
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    TrainerCallback,
)
from tqdm.auto import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProgressCallback(TrainerCallback):
    """Custom callback to display training progress with tqdm."""
    
    def __init__(self):
        super().__init__()
        self.training_bar = None
        self.epoch_bar = None
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Initialize progress bars at the start of training."""
        if state.max_steps > 0:
            self.training_bar = tqdm(
                total=state.max_steps,
                desc="Training Progress",
                unit="steps",
                dynamic_ncols=True,
                position=0,
                leave=True
            )
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Update progress at the start of each epoch."""
        if self.epoch_bar is not None:
            self.epoch_bar.close()
        
        epoch_num = int(state.epoch) if state.epoch is not None else 0
        self.epoch_bar = tqdm(
            total=state.max_steps // args.num_train_epochs,
            desc=f"Epoch {epoch_num + 1}/{args.num_train_epochs}",
            unit="steps",
            dynamic_ncols=True,
            position=1,
            leave=False
        )
    
    def on_step_end(self, args, state, control, **kwargs):
        """Update progress bars after each training step."""
        if self.training_bar is not None:
            self.training_bar.update(1)
            self.training_bar.set_postfix({
                'loss': f"{state.log_history[-1].get('loss', 0):.4f}" if state.log_history else "N/A",
                'lr': f"{state.log_history[-1].get('learning_rate', 0):.2e}" if state.log_history else "N/A"
            })
        
        if self.epoch_bar is not None:
            self.epoch_bar.update(1)
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Close epoch progress bar at the end of each epoch."""
        if self.epoch_bar is not None:
            self.epoch_bar.close()
            self.epoch_bar = None
    
    def on_train_end(self, args, state, control, **kwargs):
        """Close all progress bars at the end of training."""
        if self.training_bar is not None:
            self.training_bar.close()
            self.training_bar = None
        if self.epoch_bar is not None:
            self.epoch_bar.close()
            self.epoch_bar = None
    
    def on_evaluate(self, args, state, control, **kwargs):
        """Display evaluation message."""
        logger.info(f"Running evaluation at step {state.global_step}...")


class HateSpeechDataset(Dataset):
    """Custom dataset for multi-label hate speech classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        """
        Args:
            texts: List of text strings
            labels: List of label arrays (each array has 11 elements)
            tokenizer: Tokenizer for text encoding
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(labels, dtype=torch.float32)
        }
        
        return item


class MultiLabelRobertaWithConstraints(nn.Module):
    """
    RoBERTa model with custom head for multi-label classification with constraints.
    
    Enforces constraint: secondary labels can only be positive when primary label is 'hate'.
    Handles severe class imbalance with weighted focal loss.
    """
    
    def __init__(self, model_name, num_labels=11, dropout=0.1, class_weights=None, focal_gamma=2.0):
        super().__init__()
        
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = num_labels
        config.problem_type = "multi_label_classification"
        
        # Load base model
        self.roberta = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True
        )
        
        # Replace classifier head
        self.roberta.classifier = nn.Identity()
        
        # Custom classifier head
        hidden_size = config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        self.num_labels = num_labels
        self.focal_gamma = focal_gamma
        
        # Register class weights for imbalanced classes
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None
    
    def forward(self, input_ids, attention_mask, labels=None):
        # Get RoBERTa outputs
        outputs = self.roberta.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        sequence_output = outputs.last_hidden_state[:, 0, :]
        sequence_output = self.dropout(sequence_output)
        
        # Get logits for all labels
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            loss = self.compute_constrained_loss(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            'attentions': outputs.attentions if hasattr(outputs, 'attentions') else None,
        }
    
    def compute_constrained_loss(self, logits, labels):
        """
        Compute loss with constraint enforcement and class weighting for imbalanced labels.
        
        Constraints:
        1. Primary label (index 0): binary classification (hate/nothate)
        2. Secondary labels (indices 1-10): only valid when primary is 'hate'
        
        Implements:
        - Weighted BCE loss with class weights
        - Focal loss component to focus on hard examples
        - Constraint penalty for primary/secondary relationship
        """
        # Apply class weights if provided
        if self.class_weights is not None:
            # Use pos_weight in BCE loss (weight for positive class)
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.class_weights, reduction='none')
        else:
            loss_fct = nn.BCEWithLogitsLoss(reduction='none')
        
        # Calculate BCE loss
        bce_loss = loss_fct(logits, labels)
        
        # Focal loss component: down-weight easy examples
        # This helps the model focus on hard-to-classify examples
        probs = torch.sigmoid(logits)
        focal_weight = torch.where(
            labels == 1,
            (1 - probs) ** self.focal_gamma,  # For positive labels: weight increases when prob is low
            probs ** self.focal_gamma          # For negative labels: weight increases when prob is high
        )
        
        # Combine BCE with focal weighting
        focal_loss = (focal_weight * bce_loss).mean()
        
        # Additional constraint penalty:
        # If primary label is 0 (nothate), penalize non-zero secondary labels
        primary_labels = labels[:, 0]
        nothate_mask = (primary_labels == 0).float()
        
        # Penalty for activating secondary labels when primary is nothate
        secondary_logits = logits[:, 1:]
        secondary_probs = torch.sigmoid(secondary_logits)
        
        # Weighted penalty: penalize secondary activation for nothate samples
        constraint_penalty = (secondary_probs * nothate_mask.unsqueeze(1)).mean()
        
        # Combined loss
        total_loss = focal_loss + 0.5 * constraint_penalty
        
        return total_loss


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """
    Compute metrics for multi-label classification.
    """
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids
    
    # Apply sigmoid to get probabilities
    probs = 1 / (1 + np.exp(-predictions))
    
    # Convert to binary predictions (threshold=0.5)
    preds = (probs > 0.5).astype(int)
    
    # Enforce constraints on predictions:
    # If primary label is 0 (nothate), set all secondary labels to 0
    nothate_mask = preds[:, 0] == 0
    preds[nothate_mask, 1:] = 0
    
    # Calculate metrics
    metrics = {}
    
    # Overall metrics (micro-averaged across all labels)
    metrics['accuracy'] = accuracy_score(labels.flatten(), preds.flatten())
    metrics['f1_micro'] = f1_score(labels, preds, average='micro', zero_division=0)
    metrics['f1_macro'] = f1_score(labels, preds, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(labels, preds, average='weighted', zero_division=0)
    metrics['precision_micro'] = precision_score(labels, preds, average='micro', zero_division=0)
    metrics['recall_micro'] = recall_score(labels, preds, average='micro', zero_division=0)
    metrics['hamming_loss'] = hamming_loss(labels, preds)
    
    # Primary label metrics (hate/nothate classification)
    metrics['primary_accuracy'] = accuracy_score(labels[:, 0], preds[:, 0])
    metrics['primary_f1'] = f1_score(labels[:, 0], preds[:, 0], zero_division=0)
    metrics['primary_precision'] = precision_score(labels[:, 0], preds[:, 0], zero_division=0)
    metrics['primary_recall'] = recall_score(labels[:, 0], preds[:, 0], zero_division=0)
    
    # Secondary labels metrics (only for hate samples)
    hate_mask = labels[:, 0] == 1
    if hate_mask.sum() > 0:
        hate_labels = labels[hate_mask][:, 1:]
        hate_preds = preds[hate_mask][:, 1:]
        
        metrics['secondary_f1_micro'] = f1_score(
            hate_labels, hate_preds, average='micro', zero_division=0
        )
        metrics['secondary_f1_macro'] = f1_score(
            hate_labels, hate_preds, average='macro', zero_division=0
        )
        metrics['secondary_precision'] = precision_score(
            hate_labels, hate_preds, average='micro', zero_division=0
        )
        metrics['secondary_recall'] = recall_score(
            hate_labels, hate_preds, average='micro', zero_division=0
        )
    
    return metrics


class HateSpeechTrainer:
    """Trainer for multi-label hate speech classification."""
    
    def __init__(
        self,
        model_name: str = "unbiased-toxic-roberta",
        data_dir: str = "./data",
        output_dir: str = "./outputs",
        max_length: int = 512,
        num_labels: int = 11,
    ):
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.max_length = max_length
        self.num_labels = num_labels
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load tokenizer
        logger.info(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load label info
        self.label_info = self.load_label_info()
    
    def load_label_info(self):
        """Load label information."""
        label_info_path = self.data_dir / 'label_info.json'
        if label_info_path.exists():
            with open(label_info_path, 'r') as f:
                return json.load(f)
        return None
    
    def calculate_class_weights(self, train_df, label_names=None):
        """
        Calculate class weights for imbalanced labels.
        
        Uses custom weights for severely underrepresented labels:
        - Anti-democratic rhetoric: 10x weight
        - Praising extremist acts: 10x weight
        - Praising violence: 5x weight
        - Ideologically motivated threats: 5x weight
        - Sexual harassment: 5x weight
        - Other labels: Moderate automatic weighting (capped at 3x)
        
        Args:
            train_df: Training dataframe
            label_names: Optional list of label names for logging
        
        Returns:
            List of weights for each label (11 elements)
        """
        import ast
        
        # Parse labels from training data
        def parse_labels(label_str):
            try:
                return ast.literal_eval(label_str)
            except:
                return [0] * self.num_labels
        
        labels = train_df['labels'].apply(parse_labels).tolist()
        labels_array = np.array(labels)
        
        # Count positive samples for each label
        pos_counts = labels_array.sum(axis=0)
        total_samples = len(labels_array)
        
        # Define custom weights for severely underrepresented labels
        # Index mapping (adjust based on your actual label order):
        # 0: primary (hate/nothate)
        # 1-10: secondary labels
        custom_weights = {
            'anti-democratic rhetoric': 10.0,
            'praising extremist acts': 10.0,
            'praising violence': 5.0,
            'ideologically motivated threats': 5.0,
            'sexual harassment': 5.0,
        }
        
        weights = []
        for i in range(self.num_labels):
            pos_count = pos_counts[i]
            neg_count = total_samples - pos_count
            
            if pos_count == 0:
                # If no positive samples, use weight of 1
                weight = 1.0
            else:
                # Calculate inverse frequency
                weight = neg_count / pos_count
                
                # Cap weight at 3x for non-custom labels to prevent instability
                if i > 0:  # Skip primary label
                    # Check if this is a severely underrepresented label
                    label_name = label_names[i] if label_names and i < len(label_names) else None
                    
                    if label_name and label_name.lower() in custom_weights:
                        # Use custom weight for severely underrepresented labels
                        weight = custom_weights[label_name.lower()]
                        logger.info(f"Label {i} ({label_name}): {int(pos_count)} samples, "
                                  f"using custom weight: {weight:.2f}x")
                    else:
                        # Cap at 3x for other labels
                        weight = min(weight, 3.0)
                        if label_name:
                            logger.info(f"Label {i} ({label_name}): {int(pos_count)} samples, "
                                      f"weight: {weight:.2f}x")
                        else:
                            logger.info(f"Label {i}: {int(pos_count)} samples, weight: {weight:.2f}x")
                else:
                    # Primary label: use moderate weight
                    weight = min(weight, 2.0)
                    if label_names and i < len(label_names):
                        logger.info(f"Label {i} ({label_names[i]}): {int(pos_count)} samples, "
                                  f"weight: {weight:.2f}x")
            
            weights.append(weight)
        
        logger.info(f"Class weights calculated: {[f'{w:.2f}' for w in weights]}")
        return weights
    
    def load_datasets(self):
        """Load train, validation, and test datasets."""
        logger.info("Loading datasets...")
        
        train_df = pd.read_csv(self.data_dir / 'train.csv')
        val_df = pd.read_csv(self.data_dir / 'val.csv')
        test_df = pd.read_csv(self.data_dir / 'test.csv')
        
        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Convert labels from string representation to list
        import ast
        
        def parse_labels(label_str):
            try:
                return ast.literal_eval(label_str)
            except:
                return [0] * self.num_labels
        
        train_labels = train_df['labels'].apply(parse_labels).tolist()
        val_labels = val_df['labels'].apply(parse_labels).tolist()
        test_labels = test_df['labels'].apply(parse_labels).tolist()
        
        # Create datasets
        train_dataset = HateSpeechDataset(
            train_df['text'].tolist(),
            train_labels,
            self.tokenizer,
            self.max_length
        )
        
        val_dataset = HateSpeechDataset(
            val_df['text'].tolist(),
            val_labels,
            self.tokenizer,
            self.max_length
        )
        
        test_dataset = HateSpeechDataset(
            test_df['text'].tolist(),
            test_labels,
            self.tokenizer,
            self.max_length
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def create_model(self, class_weights=None, focal_gamma=2.0):
        """Create model with custom head."""
        logger.info(f"Creating model: {self.model_name}")
        model = MultiLabelRobertaWithConstraints(
            self.model_name,
            num_labels=self.num_labels,
            class_weights=class_weights,
            focal_gamma=focal_gamma
        )
        return model
    
    def _check_gpu_setup(self):
        """Check and log GPU configuration."""
        if torch.cuda.is_available():
            logger.info("="*60)
            logger.info("GPU Configuration")
            logger.info("="*60)
            logger.info(f"CUDA available: Yes")
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                logger.info(f"GPU {i}: {gpu_name}")
                logger.info(f"  - Compute Capability: {props.major}.{props.minor}")
                logger.info(f"  - Total Memory: {props.total_memory / 1024**3:.2f} GB")
                logger.info(f"  - Multi-Processors: {props.multi_processor_count}")
                
                # Check if it's a Blackwell GPU (compute capability 10.x)
                if props.major >= 10:
                    logger.info(f"  ⚡ Blackwell architecture detected! Optimizations enabled.")
                    logger.info(f"  💡 Recommendation: Use --bf16 for optimal performance")
            
            logger.info("="*60)
        else:
            logger.warning("⚠️  CUDA not available - training will run on CPU (slow!)")
            logger.warning("⚠️  Install CUDA-enabled PyTorch for GPU acceleration")
            logger.warning("⚠️  Run: pip install torch --index-url https://download.pytorch.org/whl/cu124")
    
    def train(
        self,
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        fp16: bool = False,
        bf16: bool = False,
        focal_gamma: float = 2.0,
    ):
        """Train the model."""
        logger.info("Starting training...")
        
        # Check GPU availability and optimize settings
        self._check_gpu_setup()
        
        # Load datasets
        train_dataset, val_dataset, test_dataset = self.load_datasets()
        
        # Calculate class weights from training data
        logger.info("Calculating class weights for imbalanced labels...")
        train_df = pd.read_csv(self.data_dir / 'train.csv')
        
        # Get label names if available
        label_names = None
        if self.label_info and 'label_names' in self.label_info:
            label_names = self.label_info['label_names']
        
        class_weights = self.calculate_class_weights(train_df, label_names)
        
        # Create model with class weights
        model = self.create_model(class_weights=class_weights, focal_gamma=focal_gamma)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            logging_dir=str(self.output_dir / 'logs'),
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_micro",
            greater_is_better=True,
            fp16=fp16 and not bf16,  # Use fp16 only if bf16 is not enabled
            bf16=bf16,  # BFloat16 - optimal for Blackwell GPUs
            report_to=["tensorboard"],
            save_total_limit=2,
            push_to_hub=False,
            disable_tqdm=False,  # Enable tqdm progress bars
            logging_first_step=True,
            # GPU optimizations
            dataloader_num_workers=0 if not torch.cuda.is_available() else 2,
            dataloader_pin_memory=torch.cuda.is_available(),
            gradient_checkpointing=False,  # Can enable for memory savings on large models
            # Advanced optimizations for modern GPUs
            torch_compile=False,  # Set to True for PyTorch 2.0+ compilation (experimental)
            optim="adamw_torch",  # Use PyTorch's fused AdamW for better performance
        )
        
        # Log training configuration
        logger.info("\nTraining Configuration:")
        logger.info(f"  Epochs: {num_epochs}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Mixed precision: {'BF16' if bf16 else 'FP16' if fp16 else 'FP32'}")
        logger.info(f"  Device: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
        if torch.cuda.is_available():
            logger.info(f"  GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        logger.info("")
        
        # Create progress callback
        progress_callback = ProgressCallback()
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[progress_callback],
        )
        
        # Train
        logger.info("Training model...")
        train_result = trainer.train()
        
        # Save model
        logger.info(f"Saving model to {self.output_dir}")
        trainer.save_model(str(self.output_dir / 'best_model'))
        self.tokenizer.save_pretrained(str(self.output_dir / 'best_model'))
        
        # Save training metrics
        with open(self.output_dir / 'train_results.json', 'w') as f:
            json.dump(train_result.metrics, f, indent=2)
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_results = trainer.evaluate(test_dataset)
        
        with open(self.output_dir / 'test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)
        
        logger.info("Training complete!")
        logger.info(f"Test F1 (micro): {test_results.get('eval_f1_micro', 0):.4f}")
        logger.info(f"Primary accuracy: {test_results.get('eval_primary_accuracy', 0):.4f}")
        
        return trainer, test_results


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune unbiased-toxic-roberta for hate speech classification")
    parser.add_argument("--model_name", type=str, default="unbiased-toxic-roberta", help="Model name or path")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (increase for better GPU utilization)")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma parameter (higher = focus more on hard examples)")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 mixed precision training")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 mixed precision (recommended for Blackwell GPUs)")
    
    args = parser.parse_args()
    
    # Validate mixed precision arguments
    if args.fp16 and args.bf16:
        logger.warning("Both --fp16 and --bf16 specified. Using BF16 (better for modern GPUs).")
        args.fp16 = False
    
    # Recommend bf16 for better precision
    if not args.fp16 and not args.bf16 and torch.cuda.is_available():
        logger.info("💡 Tip: Add --bf16 flag for faster training on modern NVIDIA GPUs (Blackwell, Hopper, etc.)")
    
    # Create trainer
    trainer = HateSpeechTrainer(
        model_name=args.model_name,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_length=args.max_length,
    )
    
    # Train
    trainer.train(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        bf16=args.bf16,
        focal_gamma=args.focal_gamma,
    )


if __name__ == "__main__":
    main()
