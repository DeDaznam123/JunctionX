"""
Script to enrich annotations_metadata.csv with multi-label classifications using OpenAI API
Similar to enrich_dataset.py but adapted for annotations_metadata.csv structure
"""

from enrich_dataset import enrich_dataset
import pandas as pd
import os

# Set your API key (or set OPENAI_API_KEY environment variable)
API_KEY = os.environ.get('OPENAI_API_KEY')

if not API_KEY:
    raise ValueError("Please set OPENAI_API_KEY environment variable")

# Define the labels to classify
labels = [
    "Incitement of Violence",
    "Praising violence",
    "Praising extremist acts",
    "Targeting ethnic or racial groups",
    "Ideologically motivated threats",
    "Anti-democratic rhetoric",
    "Personal Attacks",
    "Sexual harassment",
    "Physical violence",
    "Psychological attacks"
]

print("="*70)
print("Enriching annotations_metadata.csv with multi-label classifications")
print("="*70)

# Check if the file exists
input_file = "data/annotations_metadata.csv"
output_file = "data/annotations_metadata_enriched.csv"

if not os.path.exists(input_file):
    print(f"\nError: {input_file} not found!")
    print("Please run fix_annotations_labels.py first to add the 'text' column.")
    exit(1)

# Load and check the data
print(f"\nLoading {input_file}...")
df = pd.read_csv(input_file)
print(f"Total rows: {len(df)}")

# Check if text column exists
if 'text' not in df.columns:
    print("\nError: 'text' column not found!")
    print("Please run fix_annotations_labels.py first to add the 'text' column.")
    exit(1)

# Check for missing text
missing_text = df['text'].isna().sum()
print(f"Rows with missing text: {missing_text}")

if missing_text > 0:
    print("\nWarning: Some rows have missing text. These will be skipped during enrichment.")

print(f"\nLabel column: 'label'")
print(f"Text column: 'text'")
print(f"Labels to add: {labels}")

# Ask user for confirmation
print("\n" + "="*70)
print("Configuration:")
print("="*70)
print(f"Model: gpt-4o-mini")
print(f"Batch size: 50 rows per API call")
print(f"Max workers: 10 parallel workers")
print(f"Estimated cost: ~${len(df) * 0.0001:.2f} (approximate)")
print(f"Estimated time: ~{len(df) / 500:.1f} minutes")

# Run enrichment
print("\n" + "="*70)
print("Starting enrichment process...")
print("="*70)

df_enriched = enrich_dataset(
    input_filepath=input_file,
    output_filepath=output_file,
    text_column="text",
    label_column="label",
    new_labels=labels,
    api_key=API_KEY,
    model="gpt-5-mini",
    batch_size=50,       # Process 50 rows per API call
    max_workers=30,      # 10 parallel workers
    use_batch_api=False  # Use normal API for faster processing
)

print("\n" + "="*70)
print("Enrichment complete!")
print("="*70)

# Display statistics
print(f"\nEnriched dataset saved to: {output_file}")
print(f"Total rows processed: {len(df_enriched)}")

# Show label distribution
print("\n" + "="*70)
print("Label Distribution:")
print("="*70)
for label in labels:
    if label in df_enriched.columns:
        count = df_enriched[label].sum()
        percentage = (count / len(df_enriched)) * 100
        print(f"{label}: {int(count)}/{len(df_enriched)} ({percentage:.1f}%)")

# Show sample rows
print("\n" + "="*70)
print("Sample enriched rows:")
print("="*70)
sample_cols = ['file_id', 'label', 'text'] + labels[:3]
print(df_enriched[sample_cols].head(3).to_string())

print("\n✓ Done! Dataset enriched successfully.")
