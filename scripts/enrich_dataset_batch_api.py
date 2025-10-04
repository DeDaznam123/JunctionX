"""
Dataset Enrichment using OpenAI Batch API
50% cheaper, handles large datasets without rate limits
Best for 10,000+ rows

Author: JunctionX Team
Date: October 2025
"""

import os
import json
import time
import pandas as pd
from openai import OpenAI
from typing import List, Optional, Tuple
from tqdm import tqdm


def create_labeling_prompt(text: str, original_label: str, labels: List[str]) -> str:
    """
    Create a prompt for labeling a single text.
    
    Args:
        text: The text to analyze
        original_label: The original hate/nothate label
        labels: List of label names to evaluate
        
    Returns:
        Formatted prompt string
    """
    labels_list = "\n".join([f"- {label}" for label in labels])
    
    prompt = f"""You are analyzing text for hate speech characteristics. The text has been classified as: "{original_label}"

Text to analyze: "{text}"

For each of the following labels, determine if it applies to the text. Respond with a JSON object where each label is a key with a value of 1 (applies) or 0 (does not apply).

Labels to evaluate:
{labels_list}

Guidelines:
- Use 1 if the label clearly applies to the text
- Use 0 if the label does not apply
- Multiple labels can apply to the same text
- Consider implicit and explicit content

Respond ONLY with valid JSON in this format:
{{"label1": 0, "label2": 1, ...}}"""
    
    return prompt


def enrich_dataset_batch_api(
    input_filepath: str,
    output_filepath: str,
    text_column: str,
    label_column: str,
    new_labels: List[str],
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    row_range: Optional[Tuple[int, int]] = None,
    poll_interval: int = 60
) -> pd.DataFrame:
    """
    Enrich dataset using OpenAI Batch API (50% cheaper, async processing).
    
    Perfect for large datasets (10,000+ rows). Submits all requests as a single
    batch job that OpenAI processes asynchronously.
    
    Args:
        input_filepath: Path to input CSV file
        output_filepath: Path to save enriched CSV file
        text_column: Name of column containing text to analyze
        label_column: Name of column containing original label (hate/nothate)
        new_labels: List of new label names to add as columns
        api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        model: OpenAI model to use (default: gpt-4o-mini)
        row_range: Optional tuple (start, end) to process specific rows
                  Example: (0, 1000) processes first 1000 rows
                  Example: (3000, 3500) processes rows 3000-3499
        poll_interval: Seconds between status checks (default: 60)
        
    Returns:
        Enriched DataFrame with new label columns
        
    Time estimate:
        - 1,000 rows: ~15-30 minutes
        - 10,000 rows: ~1-2 hours
        - 40,000 rows: ~2-3 hours
        
    Cost: 50% cheaper than standard Chat Completions API
    
    Example:
        >>> labels = ["Incitement of Violence", "Racism", "Personal Attacks"]
        >>> df = enrich_dataset_batch_api(
        ...     input_filepath="HatefulData.csv",
        ...     output_filepath="HatefulData_enriched.csv",
        ...     text_column="text",
        ...     label_column="label",
        ...     new_labels=labels,
        ...     row_range=(0, 40000)  # Process 40,000 rows
        ... )
    """
    # Load API key
    if api_key is None:
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                "No API key provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
    
    client = OpenAI(api_key=api_key)
    
    # Load dataset
    print(f"\n{'='*70}")
    print(f"OpenAI Batch API Dataset Enrichment")
    print(f"{'='*70}")
    print(f"Loading dataset from {input_filepath}...")
    df = pd.read_csv(input_filepath)
    print(f"✓ Loaded {len(df)} total rows")
    
    # Apply row range
    if row_range:
        start_index, end_index = row_range
        
        # Validate row range
        if start_index < 0 or end_index > len(df):
            raise ValueError(
                f"Row range ({start_index}, {end_index}) is out of bounds. "
                f"Dataset has {len(df)} rows."
            )
        if start_index >= end_index:
            raise ValueError(
                f"Invalid row range: start ({start_index}) must be less than "
                f"end ({end_index})"
            )
        
        df = df.iloc[start_index:end_index].copy()
        print(f"✓ Row range to process: {start_index} to {end_index-1} ({len(df)} rows)")
    else:
        start_index = 0
        print(f"✓ Processing all {len(df)} rows")
    
    # Initialize label columns
    for label in new_labels:
        if label not in df.columns:
            df[label] = 0
    
    # Create batch requests
    print(f"\nCreating batch requests...")
    batch_requests = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preparing requests"):
        text = str(row[text_column])
        original_label = str(row[label_column])
        
        prompt = create_labeling_prompt(text, original_label, new_labels)
        
        batch_requests.append({
            "custom_id": f"row_{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a precise text classification assistant. Always respond with valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0,
                "response_format": {"type": "json_object"}
            }
        })
    
    print(f"✓ Created {len(batch_requests)} batch requests")
    
    # Write batch file
    batch_file_path = "batch_requests.jsonl"
    print(f"\nWriting batch file: {batch_file_path}")
    with open(batch_file_path, 'w', encoding='utf-8') as f:
        for req in batch_requests:
            f.write(json.dumps(req) + '\n')
    print(f"✓ Batch file written ({os.path.getsize(batch_file_path) / 1024:.1f} KB)")
    
    # Upload batch file
    print(f"\nUploading batch file to OpenAI...")
    with open(batch_file_path, 'rb') as f:
        batch_input_file = client.files.create(file=f, purpose='batch')
    print(f"✓ Uploaded file ID: {batch_input_file.id}")
    
    # Create batch job
    print(f"\nCreating batch job...")
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    print(f"✓ Batch created: {batch.id}")
    print(f"✓ Status: {batch.status}")
    
    # Show time estimate
    num_rows = len(df)
    if num_rows < 5000:
        est_time = "15-30 minutes"
    elif num_rows < 20000:
        est_time = "1-2 hours"
    else:
        est_time = "2-4 hours"
    
    print(f"\n{'='*70}")
    print(f"⏳ Processing {num_rows} rows...")
    print(f"⏳ Estimated time: {est_time}")
    print(f"⏳ Checking status every {poll_interval} seconds...")
    print(f"⏳ You can safely close this and check back later!")
    print(f"{'='*70}\n")
    
    # Poll for completion
    start_time = time.time()
    last_status = None
    last_progress_print = 0
    
    while True:
        batch_status = client.batches.retrieve(batch.id)
        elapsed = time.time() - start_time
        
        # Get progress info
        completed = batch_status.request_counts.completed
        total = batch_status.request_counts.total
        failed = batch_status.request_counts.failed
        in_progress = batch_status.request_counts.in_progress if hasattr(batch_status.request_counts, 'in_progress') else 0
        
        # Show progress (update every minute or when status changes)
        if (batch_status.status != last_status or 
            elapsed - last_progress_print >= 60 or 
            completed == total):
            
            progress_pct = (completed / total * 100) if total > 0 else 0
            
            print(f"[{time.strftime('%H:%M:%S')}] Status: {batch_status.status}")
            print(f"  Progress: {completed}/{total} completed ({progress_pct:.1f}%)")
            if in_progress > 0:
                print(f"  In Progress: {in_progress}")
            if failed > 0:
                print(f"  Failed: {failed}")
            print(f"  Elapsed: {elapsed/60:.1f} minutes")
            
            # Estimate remaining time
            if completed > 0 and completed < total:
                rate = completed / elapsed  # requests per second
                remaining = total - completed
                eta_seconds = remaining / rate if rate > 0 else 0
                eta_minutes = eta_seconds / 60
                
                if eta_minutes < 60:
                    print(f"  ETA: ~{eta_minutes:.0f} minutes")
                else:
                    print(f"  ETA: ~{eta_minutes/60:.1f} hours")
            
            print()
            
            last_status = batch_status.status
            last_progress_print = elapsed
        
        # Check if complete
        if batch_status.status == "completed":
            print(f"{'='*70}")
            print(f"✓ Batch completed successfully!")
            print(f"✓ Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
            print(f"✓ Completed: {completed}/{total} requests")
            if failed > 0:
                print(f"⚠️  Failed: {failed} requests")
            print(f"{'='*70}\n")
            break
        elif batch_status.status in ["failed", "expired", "cancelled"]:
            raise Exception(
                f"Batch failed with status: {batch_status.status}\n"
                f"Error: {getattr(batch_status, 'errors', 'No error details available')}"
            )
        
        time.sleep(poll_interval)
    
    # Download results
    print(f"Downloading results...")
    result_file_id = batch_status.output_file_id
    result_content = client.files.content(result_file_id)
    results = [json.loads(line) for line in result_content.text.strip().split('\n')]
    
    print(f"✓ Downloaded {len(results)} results")
    
    # Check for errors file
    if hasattr(batch_status, 'error_file_id') and batch_status.error_file_id:
        print(f"⚠️  Some requests failed. Error file ID: {batch_status.error_file_id}")
    
    # Parse results and update dataframe
    print(f"\nProcessing results...")
    errors = 0
    successful = 0
    
    for result in tqdm(results, desc="Updating dataframe"):
        try:
            custom_id = result['custom_id']
            idx = int(custom_id.split('_')[1])
            
            # Check if request was successful
            if result.get('error'):
                errors += 1
                if errors <= 5:
                    print(f"⚠️  Error in {custom_id}: {result['error']}")
                continue
            
            response_body = result['response']['body']
            content = response_body['choices'][0]['message']['content']
            labels_dict = json.loads(content)
            
            # Update dataframe
            for label in new_labels:
                if label in labels_dict:
                    df.at[idx, label] = labels_dict[label]
            
            successful += 1
            
        except Exception as e:
            errors += 1
            if errors <= 5:  # Only print first 5 errors
                print(f"⚠️  Error parsing result for {result.get('custom_id', 'unknown')}: {e}")
    
    print(f"\n✓ Successfully processed: {successful}/{len(results)} results")
    if errors > 0:
        print(f"⚠️  Errors: {errors} results could not be processed")
    
    # Save results
    print(f"\nSaving enriched dataset to {output_filepath}...")
    df.to_csv(output_filepath, index=False)
    print(f"✓ Saved successfully")
    
    # Cleanup batch file
    try:
        os.remove(batch_file_path)
        print(f"✓ Cleaned up temporary batch file")
    except:
        pass
    
    # Summary
    total_time = (time.time() - start_time) / 60
    print(f"\n{'='*70}")
    print(f"✓ ENRICHMENT COMPLETE!")
    print(f"{'='*70}")
    print(f"Total time: {total_time:.1f} minutes ({total_time/60:.2f} hours)")
    print(f"Rows processed: {len(df)}")
    print(f"New labels added: {len(new_labels)}")
    print(f"Output file: {output_filepath}")
    print(f"Cost savings: ~50% vs standard API")
    print(f"{'='*70}\n")
    
    return df


def main():
    """Command-line interface for batch API enrichment."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Enrich dataset using OpenAI Batch API (50% cheaper)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process first 10,000 rows
  python enrich_dataset_batch_api.py --input data.csv --output enriched.csv \\
      --text-column text --label-column label \\
      --labels "Violence" "Racism" "Harassment" \\
      --row-range 0 10000

  # Process all rows
  python enrich_dataset_batch_api.py --input data.csv --output enriched.csv \\
      --text-column text --label-column label \\
      --labels "Violence" "Racism" "Harassment"

Time estimates:
  1,000 rows: ~15-30 minutes
  10,000 rows: ~1-2 hours
  40,000 rows: ~2-3 hours
        """
    )
    
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    parser.add_argument("--text-column", required=True, help="Name of text column")
    parser.add_argument("--label-column", required=True, help="Name of label column")
    parser.add_argument("--labels", nargs="+", required=True, help="List of new labels to add")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model (default: gpt-4o-mini)")
    parser.add_argument("--row-range", nargs=2, type=int, metavar=("START", "END"),
                       help="Row range to process (e.g., 0 10000)")
    parser.add_argument("--poll-interval", type=int, default=60,
                       help="Seconds between status checks (default: 60)")
    
    args = parser.parse_args()
    
    try:
        # Get API key from environment
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            print("ERROR: OPENAI_API_KEY environment variable not set")
            print("Please set it using: $env:OPENAI_API_KEY='your-key-here'")
            return 1
        
        # Convert row range to tuple
        row_range = tuple(args.row_range) if args.row_range else None
        
        # Enrich dataset
        enrich_dataset_batch_api(
            input_filepath=args.input,
            output_filepath=args.output,
            text_column=args.text_column,
            label_column=args.label_column,
            new_labels=args.labels,
            api_key=api_key,
            model=args.model,
            row_range=row_range,
            poll_interval=args.poll_interval
        )
        
        return 0
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"ERROR: {str(e)}")
        print(f"{'='*70}\n")
        return 1


if __name__ == "__main__":
    exit(main())
