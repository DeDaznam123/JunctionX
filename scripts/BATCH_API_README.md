# OpenAI Batch API Integration Guide

This guide explains when and how to use the Batch API for dataset enrichment, including cost and time comparisons.

## Table of Contents
- [Overview](#overview)
- [When to Use Batch API](#when-to-use-batch-api)
- [Performance Comparison](#performance-comparison)
- [Cost Comparison](#cost-comparison)
- [Usage Examples](#usage-examples)
- [Mode Selection](#mode-selection)
- [Best Practices](#best-practices)

---

## Overview

We support two methods for enriching datasets with hate speech labels:

### 1. **Parallel Processing** (Default for <10k rows)
- **How it works**: Processes multiple API calls in parallel using ThreadPoolExecutor
- **Best for**: Small to medium datasets, testing, when you need results quickly
- **Speed**: Real-time processing, ~2-5 minutes for 1,000 rows
- **Cost**: Standard OpenAI API pricing

### 2. **Batch API** (Default for >=10k rows)
- **How it works**: Submits all requests as a single batch job that OpenAI processes asynchronously
- **Best for**: Large datasets, production runs, cost optimization
- **Speed**: Asynchronous, ~2-3 hours for 40,000 rows
- **Cost**: 50% discount on standard API pricing

---

## When to Use Batch API

### ✅ Use Batch API When:
- Processing **10,000+ rows**
- Cost savings are important (50% discount)
- You can wait 1-4 hours for results
- Running production/final enrichment
- Want to avoid rate limits
- Processing can run overnight

### ❌ Don't Use Batch API When:
- Processing **< 5,000 rows** (parallel is faster)
- Need results immediately for testing
- Iterating rapidly during development
- Time is more valuable than cost

---

## Performance Comparison

### Time Estimates by Dataset Size

| Rows | Parallel Mode | Batch API Mode | Recommended |
|------|---------------|----------------|-------------|
| 100 | ~30 seconds | ~15-20 minutes | **Parallel** ⚡ |
| 1,000 | ~2-5 minutes | ~15-30 minutes | **Parallel** ⚡ |
| 5,000 | ~10-25 minutes | ~30-60 minutes | **Parallel** ⚡ |
| 10,000 | ~20-50 minutes | ~1-2 hours | **Either** 🤔 |
| 40,000 | ~2-4 hours | ~2-3 hours | **Batch API** 💰 |
| 100,000 | ~5-10 hours | ~4-8 hours | **Batch API** 💰 |

### Processing Speed Details

**Parallel Mode:**
- Settings: `batch_size=10`, `max_workers=3`
- Processes 3 API calls simultaneously
- Each call processes 10 rows
- ~30 rows processed every few seconds
- Real-time progress tracking

**Batch API Mode:**
- Submits all rows at once
- OpenAI processes asynchronously
- No rate limits
- Checks status every 60 seconds
- Can take 15 minutes to several hours depending on OpenAI's queue

---

## Cost Comparison

### Pricing (as of October 2025)

**GPT-4o-mini:**
- Standard API: $0.15 per 1M input tokens, $0.60 per 1M output tokens
- Batch API: $0.075 per 1M input tokens, $0.30 per 1M output tokens (50% off)

### Cost Estimates

Assumptions:
- Average input: ~500 tokens per request (text + prompt + labels)
- Average output: ~150 tokens per response (JSON with labels)

| Dataset Size | Parallel Mode Cost | Batch API Cost | Savings |
|--------------|-------------------|----------------|---------|
| 1,000 rows | ~$0.05 | ~$0.025 | $0.025 |
| 10,000 rows | ~$0.50 | ~$0.25 | $0.25 |
| 40,000 rows | ~$2.00 | ~$1.00 | **$1.00** 💰 |
| 100,000 rows | ~$5.00 | ~$2.50 | **$2.50** 💰 |

**Break-even point**: For datasets with 10,000+ rows, the cost savings become significant.

---

## Usage Examples

### 1. Auto-Select Mode (Recommended)

The system automatically chooses the best mode based on dataset size:

```python
from enrich_dataset import enrich_dataset

labels = ["Violence", "Racism", "Harassment"]

# Auto-selects Parallel for <10k rows, Batch API for >=10k
df = enrich_dataset(
    input_filepath="HatefulData.csv",
    output_filepath="HatefulData_enriched.csv",
    text_column="text",
    label_column="label",
    new_labels=labels,
    row_range=(0, 40000)  # 40k rows → Batch API automatically
)
```

**Command Line:**
```bash
# Auto-selects based on row count
python enrich_dataset.py --input HatefulData.csv --output enriched.csv \
    --text-column text --label-column label \
    --labels "Violence" "Racism" "Harassment" \
    --row-range 0 40000
```

---

### 2. Force Parallel Mode (Fast)

Force parallel processing even for large datasets:

```python
# Force parallel for speed
df = enrich_dataset(
    input_filepath="HatefulData.csv",
    output_filepath="HatefulData_enriched.csv",
    text_column="text",
    label_column="label",
    new_labels=labels,
    row_range=(0, 40000),
    use_batch_api=False,  # Force parallel
    batch_size=10,
    max_workers=3
)
```

**Command Line:**
```bash
# Force parallel mode
python enrich_dataset.py --input HatefulData.csv --output enriched.csv \
    --text-column text --label-column label \
    --labels "Violence" "Racism" "Harassment" \
    --row-range 0 40000 \
    --no-batch-api  # Force parallel
```

---

### 3. Force Batch API Mode (Cost Savings)

Force Batch API even for small datasets:

```python
# Force Batch API for cost savings
df = enrich_dataset(
    input_filepath="HatefulData.csv",
    output_filepath="HatefulData_enriched.csv",
    text_column="text",
    label_column="label",
    new_labels=labels,
    row_range=(0, 1000),
    use_batch_api=True  # Force Batch API
)
```

**Command Line:**
```bash
# Force Batch API mode
python enrich_dataset.py --input HatefulData.csv --output enriched.csv \
    --text-column text --label-column label \
    --labels "Violence" "Racism" "Harassment" \
    --row-range 0 1000 \
    --use-batch-api  # Force Batch API
```

---

### 4. Direct Batch API Usage

Use the Batch API module directly for more control:

```python
from enrich_dataset_batch_api import enrich_dataset_batch_api

labels = ["Violence", "Racism", "Harassment"]

df = enrich_dataset_batch_api(
    input_filepath="HatefulData.csv",
    output_filepath="HatefulData_enriched.csv",
    text_column="text",
    label_column="label",
    new_labels=labels,
    row_range=(0, 40000),
    poll_interval=60  # Check status every 60 seconds
)
```

**Command Line:**
```bash
# Use Batch API script directly
python enrich_dataset_batch_api.py --input HatefulData.csv --output enriched.csv \
    --text-column text --label-column label \
    --labels "Violence" "Racism" "Harassment" \
    --row-range 0 40000
```

---

## Mode Selection

### Decision Tree

```
How many rows do you need to process?

├─ < 5,000 rows
│  └─ Use Parallel Mode ⚡
│     - Fastest option
│     - Results in minutes
│     - Cost difference negligible
│
├─ 5,000 - 10,000 rows
│  ├─ Need results quickly?
│  │  └─ YES → Use Parallel Mode ⚡
│  └─ Want to save money?
│     └─ YES → Use Batch API 💰
│
└─ > 10,000 rows
   ├─ Need results immediately?
   │  └─ YES → Use Parallel Mode ⚡
   │     (but expect 2-10 hours)
   └─ Can wait 2-4 hours?
      └─ YES → Use Batch API 💰
         - 50% cost savings
         - Similar or faster time
         - No rate limits
```

### Quick Reference

| Priority | Dataset Size | Recommended Mode |
|----------|-------------|------------------|
| **Speed first** | Any size | Parallel Mode |
| **Cost first** | 10k+ rows | Batch API |
| **Testing/Dev** | Any size | Parallel Mode |
| **Production** | 10k+ rows | Batch API |

---

## Best Practices

### For Parallel Mode

1. **Optimize Settings**
   ```python
   batch_size=10      # Sweet spot for speed/cost
   max_workers=3      # Don't exceed 5 to avoid rate limits
   ```

2. **Monitor Progress**
   - Shows real-time progress
   - Auto-saves every 5 batches
   - Can interrupt and resume

3. **Handle Rate Limits**
   - Default settings respect OpenAI rate limits
   - If you hit limits, reduce `max_workers`

### For Batch API

1. **Run Long Jobs Overnight**
   ```bash
   # Start before leaving for the day
   python enrich_dataset_batch_api.py --input data.csv --output enriched.csv ...
   ```

2. **Split Very Large Datasets**
   For 100,000+ rows, consider splitting:
   ```python
   # Chunk 1: rows 0-25,000
   enrich_dataset(..., row_range=(0, 25000))
   
   # Chunk 2: rows 25,000-50,000
   enrich_dataset(..., row_range=(25000, 50000))
   
   # Chunk 3: rows 50,000-75,000
   enrich_dataset(..., row_range=(50000, 75000))
   
   # Chunk 4: rows 75,000-100,000
   enrich_dataset(..., row_range=(75000, 100000))
   ```

3. **Save Batch IDs**
   - Batch API returns a batch ID
   - You can check status manually if script stops:
   ```python
   from openai import OpenAI
   client = OpenAI(api_key="your-key")
   batch = client.batches.retrieve("batch_abc123")
   print(batch.status)
   ```

4. **Monitor Costs**
   - Check usage at: https://platform.openai.com/usage
   - Batch API shows as separate line items
   - 50% discount automatically applied

### General Best Practices

1. **Always Test First**
   ```python
   # Test on 100 rows first
   enrich_dataset(..., row_range=(0, 100))
   ```

2. **Backup Original Data**
   ```bash
   cp HatefulData.csv HatefulData_backup.csv
   ```

3. **Use Row Ranges for Distributed Processing**
   ```python
   # Machine 1
   enrich_dataset(..., row_range=(0, 20000))
   
   # Machine 2
   enrich_dataset(..., row_range=(20000, 40000))
   
   # Merge later
   ```

4. **Monitor OpenAI Status**
   - Check: https://status.openai.com/
   - Batch processing may be slower during peak times

---

## Troubleshooting

### Batch API Takes Too Long

**Problem**: Batch has been processing for >4 hours

**Solutions**:
1. Check OpenAI status page for incidents
2. Peak times (US business hours) are slower
3. Try submitting during off-peak (nights/weekends)
4. Contact OpenAI support if >24 hours

### Rate Limits in Parallel Mode

**Problem**: Getting rate limit errors

**Solutions**:
1. Reduce `max_workers` to 2 or 1
2. Increase `batch_size` to reduce API calls
3. Add delays between batches
4. Switch to Batch API (no rate limits)

### Out of Memory

**Problem**: Script crashes with memory error

**Solutions**:
1. Use row ranges to process in chunks
2. Don't load entire dataset if using row_range
3. Close other applications
4. Use Batch API (lower memory usage)

---

## Summary

### Quick Decision Guide

**Choose Parallel Mode** if:
- ✅ Dataset < 10,000 rows
- ✅ Need results in minutes
- ✅ Testing/development phase
- ✅ Cost difference < $1

**Choose Batch API** if:
- ✅ Dataset > 10,000 rows
- ✅ Can wait 1-4 hours
- ✅ Production/final run
- ✅ Want 50% cost savings
- ✅ Processing very large datasets

**Auto-Select** (recommended):
```python
# Just use default - it will choose the best option!
enrich_dataset(...)  # No use_batch_api parameter
```

---

## Additional Resources

- [OpenAI Batch API Documentation](https://platform.openai.com/docs/guides/batch)
- [OpenAI Pricing](https://openai.com/pricing)
- [Project Documentation](ENRICHMENT_README.md)
- [Example Scripts](example_usage.py) and [example_usage_batch_api.py](example_usage_batch_api.py)

---

**Last Updated**: October 2025
