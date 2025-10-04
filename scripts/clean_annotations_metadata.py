"""
Script to clean annotations_metadata_enriched.csv
Sets label columns (7-16) to 0 for all rows where column 5 (label) is 'nothate'
"""

import pandas as pd

# File paths
input_file = "data/annotations_metadata_enriched.csv"
output_file = "data/annotations_metadata_enriched_cleaned.csv"

print("="*70)
print("Cleaning annotations_metadata_enriched.csv")
print("="*70)

# Load the CSV file
print(f"\nLoading {input_file}...")
df = pd.read_csv(input_file)
print(f"Total rows: {len(df)}")

# Get column names
columns = df.columns.tolist()
print(f"\nColumns: {columns}")

# Column 5 is 'label' (index 4 in 0-indexed)
label_column = columns[4]
print(f"\nLabel column (col 5): '{label_column}'")

# Columns 7-16 are indices 6-15 in 0-indexed
label_cols = columns[6:16]
print(f"Label columns (col 7-16): {label_cols}")

# Count rows where label is 'nothate'
nothate_rows = df[label_column] == 'nothate'
nothate_count = nothate_rows.sum()
print(f"\nRows with 'nothate' label: {nothate_count}")

# Count how many nothate rows have non-zero values in label columns
nothate_df = df[nothate_rows]
rows_with_labels = (nothate_df[label_cols].sum(axis=1) > 0).sum()
print(f"Nothate rows that currently have labels (will be cleared): {rows_with_labels}")

# Set columns 7-16 to 0 for all rows where label is 'nothate'
print(f"\nSetting label columns to 0 for nothate rows...")
df.loc[nothate_rows, label_cols] = 0

# Verify the fix
nothate_df_after = df[df[label_column] == 'nothate']
rows_with_labels_after = (nothate_df_after[label_cols].sum(axis=1) > 0).sum()
print(f"Nothate rows with labels after cleaning: {rows_with_labels_after}")

# Save the cleaned dataset
print(f"\nSaving cleaned dataset to {output_file}...")
df.to_csv(output_file, index=False)

print("\n" + "="*70)
print("✓ Complete!")
print("="*70)
print(f"✓ Processed {len(df)} total rows")
print(f"✓ Cleared labels for {nothate_count} nothate rows")
print(f"✓ Output saved to: {output_file}")

# Show label distribution for hate rows
print("\n" + "="*70)
print("Label Distribution (hate rows only):")
print("="*70)
hate_df = df[df[label_column] == 'hate']
print(f"Total hate rows: {len(hate_df)}")
for label in label_cols:
    count = hate_df[label].sum()
    percentage = (count / len(hate_df)) * 100 if len(hate_df) > 0 else 0
    print(f"{label}: {int(count)}/{len(hate_df)} ({percentage:.1f}%)")
