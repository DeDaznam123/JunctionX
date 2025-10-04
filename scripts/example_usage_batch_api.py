"""
Example usage of Batch API enrichment for large datasets
Best for 10,000+ rows - 50% cheaper, no rate limits

For smaller datasets (<10,000 rows), use example_usage.py with parallel processing instead.

Author: JunctionX Team
Date: October 2025
"""

from enrich_dataset_batch_api import enrich_dataset_batch_api
import os

# Set your API key (or set OPENAI_API_KEY environment variable)
# IMPORTANT: Never commit API keys to Git! Use environment variables instead
# To set in PowerShell: $env:OPENAI_API_KEY='your-key-here'
API_KEY = os.environ.get('OPENAI_API_KEY')

if not API_KEY:
    print("ERROR: OPENAI_API_KEY environment variable not set!")
    print("Please set it using: $env:OPENAI_API_KEY='your-key-here'")
    exit(1)

# Define the labels to add
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

# ============================================================================
# BATCH API - Perfect for Large Datasets!
# ============================================================================
# Benefits:
# - 50% cheaper than standard API
# - No rate limits (process as many rows as you want)
# - Fire and forget - set it and come back later
# - OpenAI handles retries automatically
#
# Time estimates:
# - 1,000 rows: ~15-30 minutes
# - 10,000 rows: ~1-2 hours  
# - 40,000 rows: ~2-3 hours
# - 100,000 rows: ~4-8 hours
#
# Cost comparison (40,000 rows):
# - Standard API: ~$2.00
# - Batch API: ~$1.00 (50% savings!)
# ============================================================================

print("\n" + "="*70)
print("Starting Batch API Enrichment")
print("="*70)
print("This will process the dataset using OpenAI's Batch API.")
print("The script will poll for completion every 60 seconds.")
print("You can safely stop and restart - batch ID will be shown.")
print("="*70 + "\n")

# Example 1: Process first 40,000 rows
df = enrich_dataset_batch_api(
    input_filepath="HatefulData.csv",
    output_filepath="HatefulData_enriched_batch.csv",
    text_column="text",
    label_column="label",
    new_labels=labels,
    api_key=API_KEY,
    model="gpt-5-nano",
    row_range=(0, 10000),  # Process first 10,000 rows
    poll_interval=60        # Check status every 60 seconds
)

print("\n" + "="*70)
print("Batch Enrichment Complete!")
print("="*70)
print(f"First few rows with new labels:")
print(df[['text', 'label'] + labels[:5]].head())
print("\n")


# ============================================================================
# More Examples
# ============================================================================

# Example 2: Process ALL rows (no row_range specified)
"""
df = enrich_dataset_batch_api(
    input_filepath="../HatefulData.csv",
    output_filepath="../HatefulData_enriched_all.csv",
    text_column="text",
    label_column="label",
    new_labels=labels,
    api_key=API_KEY,
    model="gpt-4o-mini"
    # row_range=None means process all rows
)
"""

# Example 3: Process specific range (rows 10,000-20,000)
"""
df = enrich_dataset_batch_api(
    input_filepath="../HatefulData.csv",
    output_filepath="../HatefulData_enriched_10k_20k.csv",
    text_column="text",
    label_column="label",
    new_labels=labels,
    api_key=API_KEY,
    model="gpt-4o-mini",
    row_range=(10000, 20000)
)
"""

# Example 4: Process with faster polling (check every 30 seconds)
"""
df = enrich_dataset_batch_api(
    input_filepath="../HatefulData.csv",
    output_filepath="../HatefulData_enriched_fast_poll.csv",
    text_column="text",
    label_column="label",
    new_labels=labels,
    api_key=API_KEY,
    model="gpt-4o-mini",
    row_range=(0, 5000),
    poll_interval=30  # Check more frequently
)
"""


# ============================================================================
# Tips for Large Datasets
# ============================================================================
"""
1. SPLIT LARGE DATASETS INTO CHUNKS
   For 100,000+ rows, consider processing in chunks:
   - Chunk 1: (0, 25000)
   - Chunk 2: (25000, 50000)
   - Chunk 3: (50000, 75000)
   - Chunk 4: (75000, 100000)
   
   Then merge the outputs.

2. RUN IN BACKGROUND
   On Windows PowerShell:
   Start-Job { python example_usage_batch_api.py } | Receive-Job -Wait
   
   Or use a terminal multiplexer and let it run overnight.

3. MONITOR COSTS
   Check your OpenAI usage at: https://platform.openai.com/usage
   Batch API is 50% cheaper, but still monitor spending!

4. HANDLE INTERRUPTIONS
   If the script stops, you can check batch status manually:
   
   from openai import OpenAI
   client = OpenAI(api_key="your-key")
   batch = client.batches.retrieve("batch_id_here")
   print(batch.status)
   
   Then download results when complete.

5. BACKUP YOUR DATA
   Always keep a backup of the original CSV before enrichment!
"""
