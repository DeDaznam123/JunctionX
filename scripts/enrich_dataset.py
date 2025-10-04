"""
Dataset Enrichment Script using OpenAI API
Adds nuanced multi-label classifications to hate speech datasets
"""

import pandas as pd
import json
import os
import argparse
from openai import OpenAI
from tqdm import tqdm
import time
from typing import List, Dict
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed


def load_api_key(api_key_path: str = None) -> str:
    """
    Load OpenAI API key from environment variable or file.
    
    Args:
        api_key_path: Optional path to file containing API key
        
    Returns:
        API key string
    """
    if api_key_path and os.path.exists(api_key_path):
        with open(api_key_path, 'r') as f:
            return f.read().strip()
    
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
            "or provide --api-key-path argument"
        )
    return api_key


def create_labeling_prompt(text: str, labels: List[str], original_label: str) -> str:
    """
    Create a prompt for the OpenAI API to label text with multiple categories.
    (Legacy function - kept for backward compatibility)
    
    Args:
        text: The text to analyze
        labels: List of label names to classify
        original_label: Original hate/nothate classification
        
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
- Be consistent with the original classification "{original_label}"

Respond ONLY with a valid JSON object in this exact format:
{{
    "label1": 0,
    "label2": 1,
    ...
}}"""
    
    return prompt


def create_batch_labeling_prompt(texts: List[Dict], labels: List[str]) -> str:
    """
    Create a prompt for labeling multiple texts at once in a single API call.
    
    Args:
        texts: List of dicts with 'index', 'text', and 'original_label'
        labels: List of label names to classify
        
    Returns:
        Formatted prompt string
    """
    labels_list = "\n".join([f"- {label}" for label in labels])
    
    texts_formatted = ""
    for item in texts:
        texts_formatted += f"\n[ID: {item['index']}]\nText: \"{item['text']}\"\nOriginal label: {item['original_label']}\n"
    
    prompt = f"""You are analyzing multiple texts for hate speech characteristics. 

For each text below, determine which labels apply. Respond with a JSON object where the key is the text ID and the value is an object with each label as a key with value 1 (applies) or 0 (does not apply).

Texts to analyze:
{texts_formatted}

Labels to evaluate for EACH text:
{labels_list}

Guidelines:
- Use 1 if the label clearly applies to the text
- Use 0 if the label does not apply
- Multiple labels can apply to the same text
- Consider implicit and explicit content
- Be consistent with the original classification

Respond ONLY with a valid JSON object in this EXACT format:
{{
    "0": {{"label1": 0, "label2": 1, ...}},
    "1": {{"label1": 1, "label2": 0, ...}},
    ...
}}"""
    
    return prompt


def get_labels_from_api(
    client: OpenAI,
    text: str,
    labels: List[str],
    original_label: str,
    model: str = "gpt-4o-mini",
    max_retries: int = 3
) -> Dict[str, int]:
    """
    Get multi-label classifications from OpenAI API for a single text.
    (Legacy function - kept for backward compatibility)
    
    Args:
        client: OpenAI client instance
        text: Text to classify
        labels: List of label names
        original_label: Original classification
        model: OpenAI model to use
        max_retries: Maximum number of retry attempts
        
    Returns:
        Dictionary mapping label names to binary values (0 or 1)
    """
    prompt = create_labeling_prompt(text, labels, original_label)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a precise text classification assistant. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Try to extract JSON if wrapped in markdown code blocks
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
                result_text = result_text.strip()
            
            result = json.loads(result_text)
            
            # Validate that all labels are present and values are 0 or 1
            validated_result = {}
            for label in labels:
                if label in result:
                    value = result[label]
                    validated_result[label] = 1 if value == 1 or value == "1" or value is True else 0
                else:
                    validated_result[label] = 0
            
            return validated_result
            
        except json.JSONDecodeError as e:
            if attempt == max_retries - 1:
                print(f"Failed to parse JSON after {max_retries} attempts: {e}")
                print(f"Response: {result_text}")
                return {label: 0 for label in labels}
            time.sleep(1)
            
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error calling API: {e}")
                return {label: 0 for label in labels}
            time.sleep(2)
    
    return {label: 0 for label in labels}


def get_batch_labels_from_api(
    client: OpenAI,
    texts: List[Dict],
    labels: List[str],
    model: str = "gpt-4o-mini",
    max_retries: int = 3
) -> Dict[int, Dict[str, int]]:
    """
    Get multi-label classifications for multiple texts in one API call.
    
    Args:
        client: OpenAI client instance
        texts: List of dicts with 'index', 'text', and 'original_label'
        labels: List of label names
        model: OpenAI model to use
        max_retries: Maximum number of retry attempts
        
    Returns:
        Dictionary mapping text indices to label dictionaries
    """
    prompt = create_batch_labeling_prompt(texts, labels)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a precise text classification assistant. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            result = json.loads(content)
            
            # Convert string keys to integers and validate
            parsed_result = {}
            for key, value in result.items():
                try:
                    idx = int(key)
                    # Validate that all labels are present
                    validated_labels = {}
                    for label in labels:
                        if label in value:
                            label_value = value[label]
                            validated_labels[label] = 1 if label_value == 1 or label_value == "1" or label_value is True else 0
                        else:
                            validated_labels[label] = 0
                    parsed_result[idx] = validated_labels
                except (ValueError, TypeError) as e:
                    print(f"Warning: Could not parse key '{key}' as integer: {e}")
            
            return parsed_result
            
        except json.JSONDecodeError as e:
            print(f"Attempt {attempt + 1}: JSON decode error - {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
        except Exception as e:
            print(f"Attempt {attempt + 1}: API error - {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
    
    # Return empty dict with zeros if all retries failed
    print("Warning: All API attempts failed, returning zeros for batch")
    return {item['index']: {label: 0 for label in labels} for item in texts}


def enrich_dataset(
    input_filepath: str,
    output_filepath: str,
    text_column: str,
    label_column: str,
    new_labels: List[str],
    api_key: str = None,
    model: str = "gpt-4o-mini",
    batch_size: int = 10,
    row_range: tuple = None,
    max_workers: int = 3,
    use_batch_api: bool = None,
    use_enriched_file: bool = False
) -> pd.DataFrame:
    """
    Enrich a dataset with multi-label classifications using OpenAI API.
    Supports two modes: parallel processing (fast, standard cost) or Batch API (slower, 50% cheaper).
    
    Args:
        input_filepath: Path to input CSV file
        output_filepath: Path to save enriched CSV file
        text_column: Name of column containing text
        label_column: Name of column containing original hate/nothate label
        new_labels: List of new label names to add
        api_key: OpenAI API key (optional, will use OPENAI_API_KEY env var if not provided)
        model: OpenAI model to use (default: "gpt-4o-mini")
        batch_size: Number of rows to process per API call (default: 10)
                   Only used in parallel mode. Batch API processes all at once.
        row_range: Tuple (start, end) specifying row range to enrich (e.g., (0, 100) or (3000, 3500))
                   If None, processes all rows starting from 0
        max_workers: Number of parallel workers for processing batches (default: 3)
                    Only used in parallel mode.
        use_batch_api: Whether to use OpenAI Batch API (50% cheaper, takes longer)
                      If None, automatically uses Batch API for >=10,000 rows
                      Set to False to force parallel mode (faster, standard cost)
                      Set to True to force Batch API mode (slower, 50% cheaper)
        use_enriched_file: If True, checks if label columns (12-21) contain only zeros.
                          Only processes rows where all label columns are 0 (unlabeled rows).
                          Skips rows that already have labels. (default: False)
        
    Returns:
        Enriched DataFrame
        
    Mode Selection:
        - Parallel mode (default for <10k rows): Fast, real-time processing
          Best for: Small datasets, testing, when you need results quickly
          Time: ~2-5 minutes for 1,000 rows
          Cost: Standard API pricing
          
        - Batch API mode (default for >=10k rows): Slower, 50% cheaper
          Best for: Large datasets, production runs, cost optimization
          Time: ~2-3 hours for 40,000 rows
          Cost: 50% discount on API pricing
    
    Example:
        >>> # Auto-select mode (Batch API for 40k rows)
        >>> df = enrich_dataset(
        ...     input_filepath="data.csv",
        ...     output_filepath="data_enriched.csv",
        ...     text_column="text",
        ...     label_column="label",
        ...     new_labels=["racist", "sexist", "homophobic"],
        ...     row_range=(0, 40000)
        ... )
        >>> 
        >>> # Force parallel mode for speed
        >>> df = enrich_dataset(
        ...     input_filepath="data.csv",
        ...     output_filepath="data_enriched.csv",
        ...     text_column="text",
        ...     label_column="label",
        ...     new_labels=["racist", "sexist", "homophobic"],
        ...     row_range=(0, 40000),
        ...     use_batch_api=False  # Force parallel even for large dataset
        ... )
        >>> 
        >>> # Force Batch API for cost savings
        >>> df = enrich_dataset(
        ...     input_filepath="data.csv",
        ...     output_filepath="data_enriched.csv",
        ...     text_column="text",
        ...     label_column="label",
        ...     new_labels=["racist", "sexist", "homophobic"],
        ...     row_range=(0, 1000),
        ...     use_batch_api=True  # Force batch API even for small dataset
        ... )
    """
    
    # Auto-select mode based on dataset size if not specified
    if use_batch_api is None:
        df_temp = pd.read_csv(input_filepath)
        num_rows = len(df_temp) if row_range is None else (row_range[1] - row_range[0])
        use_batch_api = num_rows >= 10000
        
        if use_batch_api:
            print(f"\nAuto-selecting Batch API mode (processing {num_rows} rows, >=10k threshold)")
            print("Batch API: 50% cheaper, takes 2-4 hours for large datasets")
            print("To force parallel mode, set use_batch_api=False\n")
        else:
            print(f"\nAuto-selecting Parallel mode (processing {num_rows} rows, <10k threshold)")
            print("Parallel mode: Fast processing, standard API cost")
            print("To force Batch API mode for cost savings, set use_batch_api=True\n")
    
    # Route to appropriate implementation
    if use_batch_api:
        # Import and use Batch API version
        try:
            from enrich_dataset_batch_api import enrich_dataset_batch_api
            return enrich_dataset_batch_api(
                input_filepath=input_filepath,
                output_filepath=output_filepath,
                text_column=text_column,
                label_column=label_column,
                new_labels=new_labels,
                api_key=api_key,
                model=model,
                row_range=row_range
            )
        except ImportError:
            print("ERROR: enrich_dataset_batch_api.py not found!")
            print("Falling back to parallel mode...")
            use_batch_api = False
    
    # Continue with parallel processing implementation below
    print("Using Parallel Processing mode (standard API)")
    print(f"Settings: batch_size={batch_size}, max_workers={max_workers}\n")
    # Load API key if not provided
    if api_key is None:
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Provide api_key parameter or set OPENAI_API_KEY environment variable"
            )
    # Load dataset
    print(f"Loading dataset from {input_filepath}...")
    df = pd.read_csv(input_filepath)
    
    # Determine row range
    if row_range is None:
        start_index = 0
        end_index = len(df)
    else:
        start_index, end_index = row_range
        if start_index < 0 or end_index > len(df) or start_index >= end_index:
            raise ValueError(f"Invalid row_range ({start_index}, {end_index}). Must be within (0, {len(df)}) and start < end")
    
    # Validate columns
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in dataset. Available columns: {df.columns.tolist()}")
    
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in dataset. Available columns: {df.columns.tolist()}")
    
    print(f"Dataset loaded: {len(df)} total rows")
    print(f"Row range to process: {start_index} to {end_index} ({end_index - start_index} rows)")
    print(f"Text column: '{text_column}'")
    print(f"Label column: '{label_column}'")
    print(f"New labels to add: {new_labels}")
    print(f"Batch size: {batch_size} rows per API call")
    print(f"Parallel workers: {max_workers}")
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Initialize new label columns if they don't exist
    for label in new_labels:
        if label not in df.columns:
            df[label] = 0  # Initialize with 0
    
    # If use_enriched_file is True, identify rows that need processing (all label columns are 0)
    rows_to_process = None
    if use_enriched_file:
        print("\n" + "="*70)
        print("Checking for rows with existing labels (use_enriched_file=True)...")
        print("="*70)
        
        # Get all label columns (assuming they're the new_labels columns)
        label_cols = [col for col in new_labels if col in df.columns]
        
        if label_cols:
            # Find rows where all label columns sum to 0
            rows_with_no_labels = df[label_cols].sum(axis=1) == 0
            rows_to_process = set(df[rows_with_no_labels].index.tolist())
            
            # Filter by row_range if specified
            if row_range is not None:
                rows_to_process = {idx for idx in rows_to_process if start_index <= idx < end_index}
            
            total_unlabeled = len(rows_to_process)
            total_labeled = (end_index - start_index) - total_unlabeled
            
            print(f"Rows with existing labels (will skip): {total_labeled}")
            print(f"Rows without labels (will process): {total_unlabeled}")
            print(f"Total rows in range: {end_index - start_index}")
            
            if total_unlabeled == 0:
                print("\n✓ All rows in range already have labels! Nothing to process.")
                df.to_csv(output_filepath, index=False)
                return df
        else:
            print("Warning: No label columns found in dataset. Processing all rows.")
            rows_to_process = None
    
    # Create all batches upfront for the specified range
    all_batches = []
    skipped_rows = 0
    for batch_start in range(start_index, end_index, batch_size):
        batch_end = min(batch_start + batch_size, end_index)
        batch_texts = []
        for idx in range(batch_start, batch_end):
            # Skip rows that already have labels if use_enriched_file is True
            if use_enriched_file and rows_to_process is not None:
                if idx not in rows_to_process:
                    skipped_rows += 1
                    continue
            
            batch_texts.append({
                'index': idx,
                'text': str(df.loc[idx, text_column]),
                'original_label': str(df.loc[idx, label_column])
            })
        
        # Only add batch if it has texts to process
        if batch_texts:
            all_batches.append((batch_start, batch_end, batch_texts))
    
    total_batches = len(all_batches)
    total_rows = end_index - start_index
    rows_to_enrich = total_rows - skipped_rows
    
    if use_enriched_file and skipped_rows > 0:
        print(f"\n✓ Skipped {skipped_rows} rows that already have labels")
        print(f"✓ Will process {rows_to_enrich} rows that need labels")
    
    if total_batches == 0:
        print("\n✓ No rows to process!")
        df.to_csv(output_filepath, index=False)
        return df
    
    print(f"\nStarting parallel enrichment for rows {start_index}-{end_index}...")
    print(f"Total batches to process: {total_batches}")
    print(f"Processing {max_workers} batches simultaneously...")
    if use_enriched_file:
        print(f"This will make {total_batches} API calls for {rows_to_enrich} unlabeled rows")
    else:
        print(f"This will make {total_batches} API calls instead of {total_rows} (single-row processing)")
    
    # Process batches in parallel
    def process_batch(batch_data):
        """Process a single batch and return results"""
        batch_start, batch_end, batch_texts = batch_data
        batch_results = get_batch_labels_from_api(
            client=client,
            texts=batch_texts,
            labels=new_labels,
            model=model
        )
        return batch_start, batch_end, batch_results
    
    batches_processed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batch jobs
        futures = {executor.submit(process_batch, batch): batch for batch in all_batches}
        
        # Process results as they complete
        for future in tqdm(as_completed(futures), total=total_batches, desc="Processing batches"):
            try:
                batch_start, batch_end, batch_results = future.result()
                
                # Update DataFrame with results
                for idx, labels_dict in batch_results.items():
                    for label, value in labels_dict.items():
                        if label in new_labels:  # Only update if it's a valid label
                            df.loc[idx, label] = value
                
                batches_processed += 1
                
                # Save progress every 5 batches
                if batches_processed % 5 == 0:
                    df.to_csv(output_filepath, index=False)
                    print(f"\nProgress saved: {batches_processed}/{total_batches} batches completed")
                    
            except Exception as e:
                batch_data = futures[future]
                print(f"\nError processing batch starting at row {batch_data[0]}: {e}")
    
    # Final save
    df.to_csv(output_filepath, index=False)
    print(f"\n✓ Enrichment complete!")
    print(f"✓ Enriched dataset saved to {output_filepath}")
    print(f"✓ Total API calls made: {batches_processed} (saved {total_rows - batches_processed} API calls!)")
    print(f"✓ Speedup from parallelization: ~{max_workers}x faster")
    
    # Print summary
    print("\n=== Summary ===")
    for label in new_labels:
        count = df[label].sum()
        percentage = (count / len(df)) * 100
        print(f"{label}: {int(count)}/{len(df)} ({percentage:.1f}%)")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Enrich hate speech dataset with nuanced multi-label classifications using OpenAI API"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output CSV file"
    )
    
    parser.add_argument(
        "--text-column",
        type=str,
        required=True,
        help="Name of the column containing text"
    )
    
    parser.add_argument(
        "--label-column",
        type=str,
        required=True,
        help="Name of the column containing original hate/nothate classification"
    )
    
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        required=True,
        help="List of new labels to add (space-separated)"
    )
    
    parser.add_argument(
        "--api-key-path",
        type=str,
        default=None,
        help="Path to file containing OpenAI API key (or set OPENAI_API_KEY env var)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of rows to process per API call (default: 10)"
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="Number of parallel workers for processing batches (default: 3)"
    )
    
    parser.add_argument(
        "--row-range",
        type=int,
        nargs=2,
        default=None,
        metavar=('START', 'END'),
        help="Row range to process: --row-range START END (e.g., --row-range 0 1000 or --row-range 3000 3500)"
    )
    
    parser.add_argument(
        "--use-batch-api",
        action="store_true",
        help="Force use of Batch API (50%% cheaper, slower). Auto-selects for >=10k rows if not specified."
    )
    
    parser.add_argument(
        "--no-batch-api",
        action="store_true",
        help="Force use of parallel processing (faster, standard cost) even for large datasets"
    )
    
    parser.add_argument(
        "--use-enriched-file",
        action="store_true",
        help="Only process rows where label columns (12-21) contain all zeros. Skip rows that already have labels."
    )
    
    args = parser.parse_args()
    
    try:
        # Load API key
        api_key = load_api_key(args.api_key_path)
        
        # Convert row_range list to tuple if provided
        row_range = tuple(args.row_range) if args.row_range else None
        
        # Determine use_batch_api based on flags
        use_batch_api = None  # Auto-select by default
        if args.use_batch_api:
            use_batch_api = True
        elif args.no_batch_api:
            use_batch_api = False
        
        # Enrich dataset
        enrich_dataset(
            input_filepath=args.input,
            output_filepath=args.output,
            text_column=args.text_column,
            label_column=args.label_column,
            new_labels=args.labels,
            api_key=api_key,
            model=args.model,
            batch_size=args.batch_size,
            row_range=row_range,
            max_workers=args.max_workers,
            use_batch_api=use_batch_api,
            use_enriched_file=args.use_enriched_file
        )
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
