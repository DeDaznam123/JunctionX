"""
Script to change 'noHate' to 'nothate' in annotations_metadata.csv
and add text content from corresponding text files
"""

import pandas as pd
import os
from tqdm import tqdm

# File paths
input_file = "data/annotations_metadata.csv"
output_file = "data/annotations_metadata.csv"  # Overwrite the original file
text_files_dir = "data/all_files"

print("="*70)
print("Fixing label values in annotations_metadata.csv")
print("="*70)

# Load the CSV file
print(f"\nLoading {input_file}...")
df = pd.read_csv(input_file)
print(f"Total rows: {len(df)}")

# Check current label values
print(f"\nCurrent label value counts:")
print(df['label'].value_counts())

# Count rows with 'noHate'
nohate_count = (df['label'] == 'noHate').sum()
print(f"\nRows with 'noHate' label: {nohate_count}")

# Change 'noHate' to 'nothate'
print(f"\nChanging 'noHate' to 'nothate'...")
df['label'] = df['label'].replace('noHate', 'nothate')

# Verify the change
print(f"\nUpdated label value counts:")
print(df['label'].value_counts())

# Verify no more 'noHate' exists
remaining_nohate = (df['label'] == 'noHate').sum()
print(f"\nRemaining 'noHate' labels: {remaining_nohate}")

# Add text content from files
print("\n" + "="*70)
print("Adding text content from files...")
print("="*70)


def read_text_file(file_id):
    """Read text from the corresponding .txt file"""
    file_path = os.path.join(text_files_dir, f"{file_id}.txt")
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        else:
            return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


# Add text column
print("\nReading text files and adding to dataframe...")
texts = []
missing_files = 0

for file_id in tqdm(df['file_id'], desc="Processing files"):
    text = read_text_file(file_id)
    if text is None:
        missing_files += 1
    texts.append(text)

df['text'] = texts

# Report statistics
print(f"\n✓ Added text column to dataframe")
print(f"✓ Successfully read {len(df) - missing_files} text files")
print(f"✗ Missing or unreadable files: {missing_files}")

if missing_files > 0:
    print(f"\nRows with missing text:")
    print(df[df['text'].isna()]['file_id'].head(10).tolist())

# Save the fixed dataset
print(f"\nSaving updated dataset to {output_file}...")
df.to_csv(output_file, index=False)

print("\n" + "="*70)
print("✓ Complete!")
print("="*70)
print(f"✓ Processed {len(df)} total rows")
print(f"✓ Changed {nohate_count} 'noHate' labels to 'nothate'")
print(f"✓ Added 'text' column with content from {len(df) - missing_files} files")
print(f"✓ Output saved to: {output_file}")

