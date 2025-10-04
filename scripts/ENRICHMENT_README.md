# Dataset Enrichment Tool

This script enriches hate speech datasets with nuanced multi-label classifications using the OpenAI API.

## Features

- **Parallel batch processing**: Process multiple batches simultaneously (3-4x faster!)
- **Batch processing**: Process multiple rows per API call (10-20x faster than single-row!)
- **Multi-label classification**: Add multiple binary labels to each text sample
- **Progress saving**: Automatically saves progress every 5 batches
- **Resume capability**: Can resume from where it left off if interrupted
- **Flexible configuration**: Customize labels, model, batch size, parallelization, and input/output files
- **Cost efficient**: Makes far fewer API calls by batching requests

## Performance

**Single-row approach (old)**:
- 38 rows = 38 API calls, sequential
- Time: ~38-114 seconds
- Cost: Standard rate × 38 calls

**Batch processing (sequential)**:
- 38 rows = 4 API calls (batch_size=10), sequential
- Time: ~12-20 seconds
- Cost: Standard rate × 4 calls

**Parallel batch processing (NEW - FASTEST!)**:
- 38 rows = 4 API calls (batch_size=10), 3 parallel workers
- Time: ~**3-6 seconds** ⚡
- Cost: Standard rate × 4 calls
- **Speedup: 6-10x faster than sequential batching, 10-20x faster than single-row!**

## Installation

Install required packages:

```powershell
pip install -r requirements_enrichment.txt
```

## Setup

### Option 1: Environment Variable
Set your OpenAI API key as an environment variable:

```powershell
$env:OPENAI_API_KEY = "sk-your-api-key-here"
```

### Option 2: API Key File
Save your API key to a text file and reference it with `--api-key-path`:

```powershell
echo "sk-your-api-key-here" > api_key.txt
```

## Usage

### Basic Usage (with parallel processing)

```powershell
python enrich_dataset.py `
  --input "../HatefulData.csv" `
  --output "../HatefulData_enriched.csv" `
  --text-column "text" `
  --label-column "label" `
  --labels racist sexist homophobic threatening violent_language `
  --batch-size 10 `
  --max-workers 3
```

### With Custom Batch Size

```powershell
python enrich_dataset.py `
  --input "../HatefulData.csv" `
  --output "../HatefulData_enriched.csv" `
  --text-column "text" `
  --label-column "label" `
  --labels racist sexist homophobic `
  --batch-size 10 `
  --max-workers 3
```

Note: 
- Adjust `batch-size` based on text length. Longer texts = smaller batch size to stay within token limits.
- Adjust `max-workers` based on your needs. More workers = faster, but may hit rate limits.

### With API Key File

```powershell
python enrich_dataset.py `
  --input "../HatefulData.csv" `
  --output "../HatefulData_enriched.csv" `
  --text-column "text" `
  --label-column "label" `
  --labels racist sexist homophobic `
  --api-key-path "api_key.txt"
```

### Using GPT-4 (more accurate but more expensive)

```powershell
python enrich_dataset.py `
  --input "../HatefulData.csv" `
  --output "../HatefulData_enriched.csv" `
  --text-column "text" `
  --label-column "label" `
  --labels racist sexist homophobic `
  --model "gpt-4o"
```

### Resume Interrupted Run

If the script is interrupted, resume from a specific row:

```powershell
python enrich_dataset.py `
  --input "../HatefulData.csv" `
  --output "../HatefulData_enriched.csv" `
  --text-column "text" `
  --label-column "label" `
  --labels racist sexist homophobic `
  --start-index 150
```

## Arguments

- `--input`: Path to input CSV file (required)
- `--output`: Path to output CSV file (required)
- `--text-column`: Name of column containing the text to analyze (required)
- `--label-column`: Name of column containing original hate/nothate label (required)
- `--labels`: Space-separated list of new labels to add (required)
- `--api-key-path`: Path to file containing API key (optional if env var is set)
- `--model`: OpenAI model to use (default: gpt-4o-mini)
- `--batch-size`: Number of rows to process per API call (default: 10)
- `--max-workers`: Number of parallel workers for processing batches (default: 3)
- `--start-index`: Row index to start from for resuming (default: 0)

## Suggested Label Categories

### Basic Categories
- `racist` - Racism or racial discrimination
- `sexist` - Sexism or gender discrimination
- `homophobic` - Homophobia or anti-LGBTQ+ content
- `transphobic` - Transphobia or anti-transgender content
- `ableist` - Ableism or disability discrimination
- `religious_hate` - Religious intolerance
- `xenophobic` - Xenophobia or anti-immigrant sentiment

### Severity & Type
- `threatening` - Contains threats or intimidation
- `violent_language` - Calls for or describes violence
- `dehumanizing` - Dehumanizes a group
- `slur_present` - Contains explicit slurs
- `implicit_hate` - Coded or implicit hate speech
- `stereotyping` - Uses harmful stereotypes

### Context
- `sarcastic` - Sarcastic or mocking tone
- `counter_speech` - Calling out hate speech
- `offensive_not_hate` - Offensive but not targeting a group
- `requires_context` - Needs additional context to judge

## Example Output

Input CSV:
```csv
text,label
"Example hateful text",hate
"Example normal text",nothate
```

Output CSV:
```csv
text,label,racist,sexist,homophobic,threatening
"Example hateful text",hate,1,0,0,1
"Example normal text",nothate,0,0,0,0
```

## Cost Estimation

With batch processing (20 rows per call):
- **gpt-4o-mini**: ~$0.15 per 1M input tokens, ~$0.60 per 1M output tokens
- **gpt-4o**: ~$2.50 per 1M input tokens, ~$10 per 1M output tokens

Approximate cost for 1000 rows with 5 labels and batch_size=20 (50 API calls):
- gpt-4o-mini: $0.05 - $0.25
- gpt-4o: $1 - $2.50

*Note: This is significantly cheaper than the old approach (1 row per call) which would cost 20x more!*

## Tips

1. **Start small**: Test on a small subset first (e.g., 10-50 rows)
2. **Choose appropriate model**: gpt-4o-mini is faster and cheaper; gpt-4o is more accurate
3. **Monitor progress**: The script saves every 10 rows by default
4. **Review results**: Manually review a sample to ensure quality
5. **Adjust labels**: Add or remove labels based on your specific needs

## Troubleshooting

**API Key Issues**:
- Ensure your API key is valid and has credits
- Check environment variable or file path is correct

**Column Not Found**:
- Verify column names match exactly (case-sensitive)
- Use `--text-column` and `--label-column` with exact names

**Rate Limiting**:
- Script includes 0.5s delay between requests
- If you hit limits, the script will retry automatically

**Resuming After Interruption**:
- Check the output file to see the last completed row
- Use `--start-index` to resume from that row
