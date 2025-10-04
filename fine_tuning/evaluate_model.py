"""
Evaluation script for the fine-tuned multi-label hate speech classification model.

Provides detailed metrics and analysis including per-label performance,
confusion matrices, and error analysis.
"""

import torch
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    hamming_loss,
    multilabel_confusion_matrix,
)
from transformers import AutoTokenizer
from train_model import MultiLabelRobertaWithConstraints, HateSpeechDataset
from tqdm.auto import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation for multi-label hate speech classification."""
    
    def __init__(
        self,
        model_path: str,
        data_dir: str = "./data",
        output_dir: str = "./evaluation_results",
        max_length: int = 512,
    ):
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.max_length = max_length
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load label info
        self.label_info = self.load_label_info()
        self.label_names = self.get_label_names()
        
        # Load model and tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = self.load_model()
    
    def load_label_info(self):
        """Load label information."""
        label_info_path = self.data_dir / 'label_info.json'
        if label_info_path.exists():
            with open(label_info_path, 'r') as f:
                return json.load(f)
        return None
    
    def get_label_names(self):
        """Get label names in order."""
        if self.label_info:
            return self.label_info['label_order']
        else:
            return ['primary_label'] + [f'label_{i}' for i in range(10)]
    
    def load_model(self):
        """Load the trained model."""
        logger.info(f"Loading model from {self.model_path}")
        
        # Try to load with the custom class
        try:
            model = MultiLabelRobertaWithConstraints(
                str(self.model_path),
                num_labels=11
            )
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def load_test_data(self):
        """Load test dataset."""
        logger.info("Loading test dataset...")
        
        test_df = pd.read_csv(self.data_dir / 'test.csv')
        
        # Parse labels
        import ast
        
        def parse_labels(label_str):
            try:
                return ast.literal_eval(label_str)
            except:
                return [0] * 11
        
        test_labels = test_df['labels'].apply(parse_labels).tolist()
        
        test_dataset = HateSpeechDataset(
            test_df['text'].tolist(),
            test_labels,
            self.tokenizer,
            self.max_length
        )
        
        return test_dataset, test_df
    
    def predict(self, dataset):
        """Generate predictions for dataset."""
        logger.info("Generating predictions...")
        
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting", unit="batch"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].cpu().numpy()
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs['logits'].cpu().numpy()
                
                # Apply sigmoid to get probabilities
                probs = 1 / (1 + np.exp(-logits))
                
                # Convert to binary predictions (threshold=0.5)
                preds = (probs > 0.5).astype(int)
                
                # Enforce constraints: if primary is 0, set all secondary to 0
                nothate_mask = preds[:, 0] == 0
                preds[nothate_mask, 1:] = 0
                
                all_predictions.append(preds)
                all_labels.append(labels)
                all_probs.append(probs)
        
        predictions = np.vstack(all_predictions)
        labels = np.vstack(all_labels)
        probs = np.vstack(all_probs)
        
        return predictions, labels, probs
    
    def compute_overall_metrics(self, predictions, labels):
        """Compute overall metrics."""
        logger.info("Computing overall metrics...")
        
        metrics = {
            'accuracy': float(accuracy_score(labels.flatten(), predictions.flatten())),
            'f1_micro': float(f1_score(labels, predictions, average='micro', zero_division=0)),
            'f1_macro': float(f1_score(labels, predictions, average='macro', zero_division=0)),
            'f1_weighted': float(f1_score(labels, predictions, average='weighted', zero_division=0)),
            'precision_micro': float(precision_score(labels, predictions, average='micro', zero_division=0)),
            'precision_macro': float(precision_score(labels, predictions, average='macro', zero_division=0)),
            'recall_micro': float(recall_score(labels, predictions, average='micro', zero_division=0)),
            'recall_macro': float(recall_score(labels, predictions, average='macro', zero_division=0)),
            'hamming_loss': float(hamming_loss(labels, predictions)),
        }
        
        return metrics
    
    def compute_per_label_metrics(self, predictions, labels):
        """Compute per-label metrics."""
        logger.info("Computing per-label metrics...")
        
        per_label_metrics = {}
        
        for i, label_name in enumerate(self.label_names):
            metrics = {
                'accuracy': float(accuracy_score(labels[:, i], predictions[:, i])),
                'f1': float(f1_score(labels[:, i], predictions[:, i], zero_division=0)),
                'precision': float(precision_score(labels[:, i], predictions[:, i], zero_division=0)),
                'recall': float(recall_score(labels[:, i], predictions[:, i], zero_division=0)),
                'support': int(labels[:, i].sum()),
            }
            per_label_metrics[label_name] = metrics
        
        return per_label_metrics
    
    def compute_hate_specific_metrics(self, predictions, labels):
        """Compute metrics specifically for hate samples."""
        logger.info("Computing hate-specific metrics...")
        
        hate_mask = labels[:, 0] == 1
        
        if hate_mask.sum() == 0:
            return {}
        
        hate_labels = labels[hate_mask][:, 1:]
        hate_preds = predictions[hate_mask][:, 1:]
        
        metrics = {
            'num_hate_samples': int(hate_mask.sum()),
            'secondary_f1_micro': float(f1_score(hate_labels, hate_preds, average='micro', zero_division=0)),
            'secondary_f1_macro': float(f1_score(hate_labels, hate_preds, average='macro', zero_division=0)),
            'secondary_precision': float(precision_score(hate_labels, hate_preds, average='micro', zero_division=0)),
            'secondary_recall': float(recall_score(hate_labels, hate_preds, average='micro', zero_division=0)),
            'secondary_hamming_loss': float(hamming_loss(hate_labels, hate_preds)),
        }
        
        return metrics
    
    def generate_confusion_matrices(self, predictions, labels):
        """Generate confusion matrices for each label."""
        logger.info("Generating confusion matrices...")
        
        confusion_matrices = {}
        
        for i, label_name in enumerate(self.label_names):
            cm = confusion_matrix(labels[:, i], predictions[:, i])
            confusion_matrices[label_name] = cm.tolist()
        
        return confusion_matrices
    
    def plot_confusion_matrix(self, cm, label_name, save_path):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive']
        )
        plt.title(f'Confusion Matrix: {label_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def error_analysis(self, predictions, labels, texts, probs):
        """Perform error analysis."""
        logger.info("Performing error analysis...")
        
        errors = []
        
        # Primary label errors (hate/nothate misclassification)
        primary_errors = predictions[:, 0] != labels[:, 0]
        
        for idx in np.where(primary_errors)[0]:
            error = {
                'index': int(idx),
                'text': texts[idx],
                'true_label': 'hate' if labels[idx, 0] == 1 else 'nothate',
                'predicted_label': 'hate' if predictions[idx, 0] == 1 else 'nothate',
                'confidence': float(probs[idx, 0]),
                'error_type': 'primary_classification',
            }
            errors.append(error)
        
        # Secondary label errors (only for correctly classified hate samples)
        hate_mask = (labels[:, 0] == 1) & (predictions[:, 0] == 1)
        
        for idx in np.where(hate_mask)[0]:
            secondary_errors = predictions[idx, 1:] != labels[idx, 1:]
            
            if secondary_errors.any():
                error = {
                    'index': int(idx),
                    'text': texts[idx],
                    'true_secondary': labels[idx, 1:].tolist(),
                    'predicted_secondary': predictions[idx, 1:].tolist(),
                    'error_type': 'secondary_classification',
                }
                errors.append(error)
        
        return errors
    
    def evaluate(self):
        """Run complete evaluation pipeline."""
        logger.info("="*60)
        logger.info("STARTING MODEL EVALUATION")
        logger.info("="*60)
        
        # Load test data
        test_dataset, test_df = self.load_test_data()
        
        # Generate predictions
        predictions, labels, probs = self.predict(test_dataset)
        
        # Compute metrics
        overall_metrics = self.compute_overall_metrics(predictions, labels)
        per_label_metrics = self.compute_per_label_metrics(predictions, labels)
        hate_metrics = self.compute_hate_specific_metrics(predictions, labels)
        confusion_matrices = self.generate_confusion_matrices(predictions, labels)
        
        # Error analysis
        errors = self.error_analysis(
            predictions,
            labels,
            test_df['text'].tolist(),
            probs
        )
        
        # Save results
        results = {
            'overall_metrics': overall_metrics,
            'per_label_metrics': per_label_metrics,
            'hate_specific_metrics': hate_metrics,
            'num_samples': len(predictions),
            'num_errors': len(errors),
        }
        
        with open(self.output_dir / 'evaluation_metrics.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        with open(self.output_dir / 'confusion_matrices.json', 'w') as f:
            json.dump(confusion_matrices, f, indent=2)
        
        # Save first 100 errors for review
        errors_sample = errors[:100]
        with open(self.output_dir / 'error_samples.json', 'w') as f:
            json.dump(errors_sample, f, indent=2)
        
        # Plot confusion matrices
        plots_dir = self.output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        for i, label_name in enumerate(self.label_names):
            cm = np.array(confusion_matrices[label_name])
            save_path = plots_dir / f'confusion_matrix_{label_name}.png'
            self.plot_confusion_matrix(cm, label_name, save_path)
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Total samples: {len(predictions)}")
        logger.info(f"Overall F1 (micro): {overall_metrics['f1_micro']:.4f}")
        logger.info(f"Overall F1 (macro): {overall_metrics['f1_macro']:.4f}")
        logger.info(f"Primary accuracy: {per_label_metrics['primary_label']['accuracy']:.4f}")
        logger.info(f"Primary F1: {per_label_metrics['primary_label']['f1']:.4f}")
        
        if hate_metrics:
            logger.info(f"\nHate samples: {hate_metrics['num_hate_samples']}")
            logger.info(f"Secondary F1 (micro): {hate_metrics['secondary_f1_micro']:.4f}")
            logger.info(f"Secondary F1 (macro): {hate_metrics['secondary_f1_macro']:.4f}")
        
        logger.info(f"\nTotal errors: {len(errors)}")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info("="*60 + "\n")
        
        return results


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned hate speech classification model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="Output directory")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_length=args.max_length,
    )
    
    # Run evaluation
    results = evaluator.evaluate()


if __name__ == "__main__":
    main()
