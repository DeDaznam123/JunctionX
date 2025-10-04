"""
Data preparation script for fine-tuning unbiased-toxic-roberta model.
Combines annotations_data_final.csv and HatefulData_enriched_fixed.csv for multi-label classification.

Label structure:
- Primary label: hate/nothate (mutually exclusive)
- Secondary labels (only for hate): 10 categories
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import json


class DataPreparator:
    """Prepares and validates data for multi-label hate speech classification."""
    
    # 10 secondary labels (only applicable when primary label is 'hate')
    SECONDARY_LABELS = [
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
    
    def __init__(self, data_dir='../data'):
        """Initialize with data directory path."""
        self.data_dir = Path(data_dir)
        self.df_combined = None
        
    def load_data(self):
        """Load both CSV files."""
        print("Loading datasets...")
        
        # Load annotations_data_final.csv
        df_annotations = pd.read_csv(
            self.data_dir / 'annotations_data_final.csv'
        )
        print(f"Loaded annotations_data_final.csv: {len(df_annotations)} rows")
        
        # Load HatefulData_enriched_fixed.csv
        df_hateful = pd.read_csv(
            self.data_dir / 'HatefulData_enriched_fixed.csv'
        )
        print(f"Loaded HatefulData_enriched_fixed.csv: {len(df_hateful)} rows")
        
        return df_annotations, df_hateful
    
    def standardize_columns(self, df_annotations, df_hateful):
        """Standardize column names and select relevant columns."""
        print("\nStandardizing data format...")
        
        # For annotations_data_final: already has correct format
        cols_to_keep = ['text', 'label'] + self.SECONDARY_LABELS
        df_ann_clean = df_annotations[cols_to_keep].copy()
        
        # For HatefulData: select and rename columns
        cols_hateful = ['text', 'label'] + self.SECONDARY_LABELS
        df_hate_clean = df_hateful[cols_hateful].copy()
        
        return df_ann_clean, df_hate_clean
    
    def validate_data_integrity(self, df):
        """Validate data integrity and label constraints."""
        print("\nValidating data integrity...")
        
        # Check for missing values in text or label
        missing_text = df['text'].isna().sum()
        missing_label = df['label'].isna().sum()
        print(f"Missing text values: {missing_text}")
        print(f"Missing label values: {missing_label}")
        
        # Remove rows with missing text or label
        df_clean = df.dropna(subset=['text', 'label']).copy()
        
        # Validate label values
        valid_labels = df_clean['label'].isin(['hate', 'nothate'])
        invalid_count = (~valid_labels).sum()
        if invalid_count > 0:
            print(f"WARNING: Found {invalid_count} invalid labels. Removing...")
            df_clean = df_clean[valid_labels].copy()
        
        # Validate constraint: nothate entries should have all secondary labels as 0
        nothate_mask = df_clean['label'] == 'nothate'
        nothate_rows = df_clean[nothate_mask]
        
        for col in self.SECONDARY_LABELS:
            # Convert to numeric, handling any non-numeric values
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype(int)
            
            violations = (nothate_rows[col] != 0).sum()
            if violations > 0:
                print(f"WARNING: {violations} 'nothate' entries have non-zero '{col}'. Fixing...")
                df_clean.loc[nothate_mask, col] = 0
        
        # Ensure all secondary labels are 0 or 1
        for col in self.SECONDARY_LABELS:
            df_clean[col] = df_clean[col].clip(0, 1).astype(int)
        
        print(f"Clean dataset size: {len(df_clean)} rows")
        return df_clean
    
    def create_label_encoding(self, df):
        """Create encoded labels for model training."""
        print("\nCreating label encodings...")
        
        # Primary label: hate=1, nothate=0
        df['primary_label'] = (df['label'] == 'hate').astype(int)
        
        # Secondary labels are already 0/1 encoded
        # Create a combined multi-label array: [primary_label, ...secondary_labels]
        label_columns = ['primary_label'] + self.SECONDARY_LABELS
        df['labels'] = df[label_columns].values.tolist()
        
        return df
    
    def get_dataset_statistics(self, df):
        """Calculate and display dataset statistics."""
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        
        total_samples = len(df)
        hate_samples = (df['primary_label'] == 1).sum()
        nothate_samples = (df['primary_label'] == 0).sum()
        
        print(f"\nTotal samples: {total_samples}")
        print(f"Hate samples: {hate_samples} ({hate_samples/total_samples*100:.2f}%)")
        print(f"Not-hate samples: {nothate_samples} ({nothate_samples/total_samples*100:.2f}%)")
        
        print("\nSecondary label distribution (hate samples only):")
        hate_df = df[df['primary_label'] == 1]
        for label in self.SECONDARY_LABELS:
            count = hate_df[label].sum()
            pct = count / len(hate_df) * 100 if len(hate_df) > 0 else 0
            print(f"  {label}: {count} ({pct:.2f}%)")
        
        # Average number of secondary labels per hate sample
        if len(hate_df) > 0:
            avg_labels = hate_df[self.SECONDARY_LABELS].sum(axis=1).mean()
            print(f"\nAverage secondary labels per hate sample: {avg_labels:.2f}")
        
        print("="*60 + "\n")
    
    def split_data(self, df, test_size=0.15, val_size=0.15, random_state=42):
        """Split data into train, validation, and test sets."""
        print(f"\nSplitting data (train/val/test)...")
        
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df['primary_label']
        )
        
        # Second split: separate validation from training
        val_relative_size = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_relative_size,
            random_state=random_state,
            stratify=train_val_df['primary_label']
        )
        
        print(f"Train set: {len(train_df)} samples")
        print(f"Validation set: {len(val_df)} samples")
        print(f"Test set: {len(test_df)} samples")
        
        return train_df, val_df, test_df
    
    def save_processed_data(self, train_df, val_df, test_df, output_dir='../fine_tuning/data'):
        """Save processed datasets to CSV and JSON files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving processed data to {output_path}...")
        
        # Save as CSV
        train_df.to_csv(output_path / 'train.csv', index=False)
        val_df.to_csv(output_path / 'val.csv', index=False)
        test_df.to_csv(output_path / 'test.csv', index=False)
        
        # Save label information
        label_info = {
            'primary_label': {
                'name': 'primary_label',
                'type': 'binary',
                'classes': ['nothate', 'hate'],
                'description': 'Primary classification: hate or not-hate (mutually exclusive)'
            },
            'secondary_labels': {
                'names': self.SECONDARY_LABELS,
                'type': 'multi-label',
                'description': 'Secondary labels only applicable when primary_label is hate',
                'constraint': 'All secondary labels must be 0 when primary_label is nothate'
            },
            'total_labels': 11,  # 1 primary + 10 secondary
            'label_order': ['primary_label'] + self.SECONDARY_LABELS
        }
        
        with open(output_path / 'label_info.json', 'w') as f:
            json.dump(label_info, f, indent=2)
        
        print("Data saved successfully!")
        print(f"  - train.csv: {len(train_df)} samples")
        print(f"  - val.csv: {len(val_df)} samples")
        print(f"  - test.csv: {len(test_df)} samples")
        print(f"  - label_info.json: Label configuration")
    
    def prepare(self):
        """Main pipeline to prepare data."""
        print("="*60)
        print("STARTING DATA PREPARATION PIPELINE")
        print("="*60)
        
        # Load data
        df_annotations, df_hateful = self.load_data()
        
        # Standardize columns
        df_ann_clean, df_hate_clean = self.standardize_columns(df_annotations, df_hateful)
        
        # Combine datasets
        print(f"\nCombining datasets...")
        df_combined = pd.concat([df_ann_clean, df_hate_clean], ignore_index=True)
        print(f"Combined dataset: {len(df_combined)} rows")
        
        # Validate data integrity
        df_clean = self.validate_data_integrity(df_combined)
        
        # Create label encodings
        df_encoded = self.create_label_encoding(df_clean)
        
        # Show statistics
        self.get_dataset_statistics(df_encoded)
        
        # Split data
        train_df, val_df, test_df = self.split_data(df_encoded)
        
        # Save processed data
        self.save_processed_data(train_df, val_df, test_df)
        
        print("\n" + "="*60)
        print("DATA PREPARATION COMPLETE!")
        print("="*60)
        
        return train_df, val_df, test_df


if __name__ == "__main__":
    preparator = DataPreparator(data_dir='../data')
    train_df, val_df, test_df = preparator.prepare()
