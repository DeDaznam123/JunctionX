"""
Script to set label columns (12-21) to 0 for all rows where column 4 (label) is 'nothate'
"""

import pandas as pd

# File paths
input_file = "HatefulData_re_enriched.csv"
output_file = "HatefulData_re_enriched_fixed.csv"

print("="*70)
print("Fixing nothate labels")
print("="*70)

# Load the CSV file
print(f"\nLoading {input_file}...")
df = pd.read_csv(input_file)
print(f"Total rows: {len(df)}")

# Get column names
columns = df.columns.tolist()
print(f"\nColumns: {columns[:15]}...")  # Show first 15 columns

# Column 4 is 'label' (index 3 in 0-indexed)
label_column = columns[3]
print(f"\nLabel column (col 4): '{label_column}'")

# Columns 12-21 are indices 11-20 in 0-indexed (inclusive)
label_cols = columns[11:21]
print(f"Label columns (col 12-21): {label_cols}")

# Count rows where label is 'nothate'
nothate_rows = df[label_column] == 'nothate'
nothate_count = nothate_rows.sum()
print(f"\nRows with 'nothate' label: {nothate_count}")

# Count how many nothate rows have non-zero values in label columns
nothate_df = df[nothate_rows]
rows_with_labels = (nothate_df[label_cols].sum(axis=1) > 0).sum()
print(f"Nothate rows that currently have labels (will be cleared): {rows_with_labels}")

# Set columns 12-21 to 0 for all rows where label is 'nothate'
print(f"\nSetting label columns to 0 for nothate rows...")
df.loc[nothate_rows, label_cols] = 0

# Verify the fix
nothate_df_after = df[df[label_column] == 'nothate']
rows_with_labels_after = (nothate_df_after[label_cols].sum(axis=1) > 0).sum()
print(f"Nothate rows with labels after fix: {rows_with_labels_after}")

# Save the fixed dataset
print(f"\nSaving fixed dataset to {output_file}...")
df.to_csv(output_file, index=False)

print("\n" + "="*70)
print("✓ Complete!")
print("="*70)
print(f"✓ Processed {len(df)} total rows")
print(f"✓ Cleared labels for {nothate_count} nothate rows")
print(f"✓ Output saved to: {output_file}")
