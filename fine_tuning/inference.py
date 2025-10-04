"""
Inference script for making predictions with the fine-tuned model.

Usage:
    python inference.py --model_path ./outputs/best_model --text "Your text here"
    python inference.py --model_path ./outputs/best_model --input_file texts.txt
"""

import torch
import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer
from train_model import MultiLabelRobertaWithConstraints
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HateSpeechPredictor:
    """Predictor for hate speech classification."""
    
    LABEL_NAMES = [
        'primary_label',
        'Incitement of Violence',
        'Praising violence',
        'Praising extremist acts',
        'Targeting ethnic or racial groups',
        'Ideologically motivated threats',
        'Anti-democratic rhetoric',
        'Personal Attacks',
        'Sexual harassment',
        'Physical violence',
        'Psychological attacks'
    ]
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to the trained model directory
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_path = Path(model_path)
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        
        # Load model
        logger.info(f"Loading model from {self.model_path}")
        self.model = MultiLabelRobertaWithConstraints(
            str(self.model_path),
            num_labels=11
        )
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Model loaded successfully!")
    
    def predict_single(self, text: str, return_probs: bool = False):
        """
        Predict labels for a single text.
        
        Args:
            text: Input text to classify
            return_probs: If True, return probabilities instead of binary predictions
        
        Returns:
            Dictionary with predictions
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs['logits'].cpu().numpy()[0]
        
        # Apply sigmoid to get probabilities
        probs = 1 / (1 + torch.exp(-torch.tensor(logits))).numpy()
        
        # Convert to binary predictions
        preds = (probs > 0.5).astype(int)
        
        # Enforce constraint: if primary is 0 (nothate), set all secondary to 0
        if preds[0] == 0:
            preds[1:] = 0
            if not return_probs:
                probs[1:] = 0  # Also zero out probabilities for consistency
        
        # Build result
        result = {
            'text': text,
            'primary_classification': 'hate' if preds[0] == 1 else 'nothate',
            'primary_confidence': float(probs[0]),
        }
        
        if return_probs:
            result['label_probabilities'] = {
                self.LABEL_NAMES[i]: float(probs[i]) 
                for i in range(len(self.LABEL_NAMES))
            }
        else:
            result['label_predictions'] = {
                self.LABEL_NAMES[i]: int(preds[i]) 
                for i in range(len(self.LABEL_NAMES))
            }
        
        # Add active secondary labels (only for hate)
        if preds[0] == 1:
            active_labels = [
                self.LABEL_NAMES[i] 
                for i in range(1, len(self.LABEL_NAMES)) 
                if preds[i] == 1
            ]
            result['secondary_labels'] = active_labels
        else:
            result['secondary_labels'] = []
        
        return result
    
    def predict_batch(self, texts: list, return_probs: bool = False):
        """
        Predict labels for multiple texts.
        
        Args:
            texts: List of input texts to classify
            return_probs: If True, return probabilities instead of binary predictions
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        logger.info(f"Processing {len(texts)} texts...")
        
        for i, text in enumerate(texts):
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(texts)} texts")
            
            result = self.predict_single(text, return_probs=return_probs)
            results.append(result)
        
        logger.info(f"Completed processing {len(texts)} texts")
        
        return results
    
    def predict_from_file(self, input_file: str, output_file: str = None, return_probs: bool = False):
        """
        Predict labels for texts in a file (one text per line).
        
        Args:
            input_file: Path to input file with texts (one per line)
            output_file: Path to save results (JSON format). If None, prints to console.
            return_probs: If True, return probabilities instead of binary predictions
        """
        # Read texts
        with open(input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Loaded {len(texts)} texts from {input_file}")
        
        # Get predictions
        results = self.predict_batch(texts, return_probs=return_probs)
        
        # Save or print results
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {output_file}")
        else:
            print(json.dumps(results, indent=2, ensure_ascii=False))
        
        return results


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description="Make predictions with fine-tuned hate speech model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--text", type=str, help="Single text to classify")
    parser.add_argument("--input_file", type=str, help="File with texts to classify (one per line)")
    parser.add_argument("--output_file", type=str, help="Output file for results (JSON)")
    parser.add_argument("--return_probs", action="store_true", help="Return probabilities instead of binary predictions")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.text and not args.input_file:
        parser.error("Either --text or --input_file must be provided")
    
    # Create predictor
    predictor = HateSpeechPredictor(args.model_path, device=args.device)
    
    # Make predictions
    if args.text:
        # Single text prediction
        result = predictor.predict_single(args.text, return_probs=args.return_probs)
        print("\n" + "="*60)
        print("PREDICTION RESULT")
        print("="*60)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("="*60 + "\n")
    
    elif args.input_file:
        # Batch prediction from file
        predictor.predict_from_file(
            args.input_file,
            args.output_file,
            return_probs=args.return_probs
        )


if __name__ == "__main__":
    main()
