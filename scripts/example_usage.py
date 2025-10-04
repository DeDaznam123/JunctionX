"""
Example usage of enrich_dataset function with parallel batch processing
Re-process rows that have no labels (all 0s in label columns)
"""

from enrich_dataset import enrich_dataset
import pandas as pd
import os

# Set your API key (or set OPENAI_API_KEY environment variable)
# IMPORTANT: Never commit API keys to Git! Use environment variables instead
API_KEY = os.environ.get('OPENAI_API_KEY')

if not API_KEY:
    raise ValueError("Please set OPENAI_API_KEY environment variable")

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

# Step 1: Load the enriched dataset and check status
print("="*70)
print("Processing enriched dataset...")
print("="*70)

# Step 2: Re-enrich rows using use_enriched_file mode
print("\n" + "="*70)
print("Re-enriching unlabeled rows...")
print("="*70)

df_enriched = enrich_dataset(
    input_filepath="HatefulData_enriched.csv",
    output_filepath="HatefulData_enriched_updated.csv",
    text_column="text",
    label_column="label",
    new_labels=labels,
    api_key=API_KEY,
    model="gpt-5-mini",
    row_range=(40000, 40623),  # Process rows in this range
    batch_size=60,       # Process 60 rows per API call
    max_workers=40,      # 20 parallel workers
    use_batch_api=False,  # Use normal OpenAI API (parallel mode) instead of Batch API
    use_enriched_file=True  # Automatically skip rows that already have labels
)

print("\n" + "="*70)
print("Re-enrichment complete!")
print("="*70)
print(f"\nFirst few rows:")
print(df_enriched[['text', 'label'] + labels[:5]].head())

# Verify no more rows with all 0s in the processed range
rows_in_range = df_enriched.iloc[40000:40623]
remaining_unlabeled = (rows_in_range[labels].sum(axis=1) == 0).sum()
print(f"\nRemaining rows with all 0s in range [20000-30000]: {remaining_unlabeled}")
print(f"✓ Total rows in dataset: {len(df_enriched)}")


