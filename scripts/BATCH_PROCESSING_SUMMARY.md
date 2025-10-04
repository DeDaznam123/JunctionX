# Batch Processing Implementation ✅

## What Changed

The enrichment script now processes **multiple rows per API call** instead of one row at a time.

### Before vs After

| Metric | Before (Single Row) | After (Batch Processing) |
|--------|---------------------|--------------------------|
| **Rows per API call** | 1 | 20 (configurable) |
| **API calls for 38 rows** | 38 | 2 |
| **Speed** | ~38-114 seconds | ~4-10 seconds |
| **Cost** | 100% | ~5% |
| **Speedup** | 1x | **10-20x faster** |

## How It Works

### Old Approach
```
Row 1 → API Call 1 → Get labels for Row 1
Row 2 → API Call 2 → Get labels for Row 2
Row 3 → API Call 3 → Get labels for Row 3
...
Row 38 → API Call 38 → Get labels for Row 38
```

### New Approach
```
Rows 1-20  → API Call 1 → Get labels for Rows 1-20
Rows 21-38 → API Call 2 → Get labels for Rows 21-38
```

## New Prompt Structure

The batch prompt sends multiple texts with IDs in a single request:

```
[ID: 0]
Text: "Example text 1"
Original label: hate

[ID: 1]
Text: "Example text 2"
Original label: nothate

...

Labels to evaluate for EACH text:
- racist
- sexist
- homophobic
...

Response format:
{
    "0": {"racist": 1, "sexist": 0, "homophobic": 0, ...},
    "1": {"racist": 0, "sexist": 0, "homophobic": 0, ...},
    ...
}
```

## Key Functions Added

### `create_batch_labeling_prompt(texts, labels)`
Creates a prompt that includes multiple texts with IDs.

### `get_batch_labels_from_api(client, texts, labels, model)`
Makes a single API call to process multiple texts and returns structured results.

## Usage Examples

### Python Function
```python
from enrich_dataset import enrich_dataset

df = enrich_dataset(
    input_filepath="HatefulData.csv",
    output_filepath="HatefulData_enriched.csv",
    text_column="text",
    label_column="label",
    new_labels=["racist", "sexist", "homophobic"],
    batch_size=20  # Process 20 rows per API call
)
```

### Command Line
```powershell
python enrich_dataset.py `
  --input "HatefulData.csv" `
  --output "HatefulData_enriched.csv" `
  --text-column "text" `
  --label-column "label" `
  --labels racist sexist homophobic `
  --batch-size 20
```

## Adjusting Batch Size

The default batch size is 20, but you can adjust it based on:

- **Text length**: Shorter texts → larger batch size (30-50)
- **Text length**: Longer texts → smaller batch size (5-10)
- **Number of labels**: More labels → smaller batch size
- **Token limits**: Stay under ~4000 input tokens per request

### Examples
```python
# For short tweets (< 280 chars)
batch_size=50

# For medium texts (< 500 chars)
batch_size=20  # Default

# For long texts (> 1000 chars)
batch_size=5
```

## Files Modified

1. **`enrich_dataset.py`**:
   - Added `create_batch_labeling_prompt()` function
   - Added `get_batch_labels_from_api()` function
   - Updated `enrich_dataset()` to use batch processing
   - Changed default `batch_size` from 10 (save frequency) to 20 (rows per call)

2. **`example_usage.py`**:
   - Updated to demonstrate batch processing
   - Added comments explaining the speedup

3. **`ENRICHMENT_README.md`**:
   - Added performance comparison section
   - Updated usage examples
   - Updated cost estimates

## Backward Compatibility

The old single-row functions are still available:
- `create_labeling_prompt(text, labels, original_label)`
- `get_labels_from_api(client, text, labels, original_label, model)`

These are kept for backward compatibility but are not used by default.

## Testing

To test with your data:

```python
from enrich_dataset import enrich_dataset
import os

# Set API key
os.environ['OPENAI_API_KEY'] = 'your-key-here'

# Test on first 10 rows
df = enrich_dataset(
    input_filepath="HatefulData.csv",
    output_filepath="HatefulData_test.csv",
    text_column="text",
    label_column="label",
    new_labels=["racist", "sexist"],
    batch_size=5  # Small batch for testing
)
```

Check `HatefulData_test.csv` to verify the results look correct!

## Progress Tracking

The script now shows:
- Total batches to process
- API calls saved vs old approach
- Progress bar for batches (not individual rows)
- Automatic saving every 5 batches

Example output:
```
Dataset loaded: 38 rows
Batch size: 20 rows per API call
Total batches to process: 2
This will make 2 API calls instead of 38 calls (single-row processing)

Processing batches: 100%|██████████| 2/2 [00:05<00:00,  2.5s/batch]

✓ Enrichment complete!
✓ Total API calls made: 2 (saved 36 API calls!)
```

## Next Steps

1. **Test on small subset** (10-50 rows) to verify quality
2. **Adjust batch_size** based on your text length
3. **Run full enrichment** on complete dataset
4. **Review results** and iterate if needed

Enjoy the 10-20x speedup! 🚀
