import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np

class HateSpeechClassifier:
    """Hate Speech Classifier using fine-tuned unbiased-toxic-roberta"""
    
    def __init__(self, model_path):
        """
        Initialize the classifier with a trained model
        
        Args:
            model_path: Path to the fine-tuned model directory
        """
        print(f"Loading model from {model_path}...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully!")
    
    def predict(self, text, return_probabilities=False):
        """
        Predict hate speech for a single text
        
        Args:
            text: Input text string
            return_probabilities: If True, return probabilities for both classes
            
        Returns:
            prediction: 'hate' or 'nothate'
            probabilities (optional): dict with probabilities for each class
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=128,
            padding='max_length'
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
        
        label = 'hate' if prediction == 1 else 'nothate'
        
        if return_probabilities:
            probs = {
                'nothate': probabilities[0][0].item(),
                'hate': probabilities[0][1].item()
            }
            return label, probs
        
        return label
    
    def predict_batch(self, texts, batch_size=16):
        """
        Predict hate speech for multiple texts
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            
        Returns:
            predictions: List of predictions ('hate' or 'nothate')
            probabilities: List of probability dicts
        """
        predictions = []
        probabilities = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors='pt',
                truncation=True,
                max_length=128,
                padding='max_length'
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                batch_probs = torch.nn.functional.softmax(logits, dim=1)
                batch_preds = torch.argmax(batch_probs, dim=1)
            
            # Convert to labels
            for pred, probs in zip(batch_preds, batch_probs):
                label = 'hate' if pred.item() == 1 else 'nothate'
                predictions.append(label)
                probabilities.append({
                    'nothate': probs[0].item(),
                    'hate': probs[1].item()
                })
        
        return predictions, probabilities


def main():
    """Example usage of the HateSpeechClassifier"""
    
    # Path to your fine-tuned model
    MODEL_PATH = r"c:\Users\mitev\DELFT\JunctionX\models\hate_speech_model"
    
    # Initialize classifier
    classifier = HateSpeechClassifier(MODEL_PATH)
    
    # Example texts to classify
    example_texts = [
        "I love spending time with my friends",
        "Women should not be allowed to vote",
        "The weather is nice today",
        "All immigrants are criminals",
        "I enjoy reading books in the park"
    ]
    
    print("\n" + "="*70)
    print("Single Text Predictions:")
    print("="*70 + "\n")
    
    for text in example_texts:
        prediction, probs = classifier.predict(text, return_probabilities=True)
        print(f"Text: {text}")
        print(f"Prediction: {prediction}")
        print(f"Probabilities: nothate={probs['nothate']:.4f}, hate={probs['hate']:.4f}")
        print("-" * 70)
    
    # Batch prediction example
    print("\n" + "="*70)
    print("Batch Predictions:")
    print("="*70 + "\n")
    
    predictions, probabilities = classifier.predict_batch(example_texts)
    
    results_df = pd.DataFrame({
        'text': example_texts,
        'prediction': predictions,
        'prob_nothate': [p['nothate'] for p in probabilities],
        'prob_hate': [p['hate'] for p in probabilities]
    })
    
    print(results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv('predictions.csv', index=False)
    print("\nPredictions saved to predictions.csv")


if __name__ == "__main__":
    main()
