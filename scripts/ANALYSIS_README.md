# Enriched Data Analysis Script

This script analyzes the enriched hate speech dataset, counting label frequencies, finding co-occurrences, and creating comprehensive visualizations.

## What It Does

1. **Counts individual label frequencies** - How many times each label appears
2. **Analyzes label co-occurrences** - Which labels appear together
3. **Examines labels per text** - Distribution of how many labels each text has
4. **Compares hate vs non-hate texts** - How labels differ between categories
5. **Creates detailed visualizations** - Multiple charts and heatmaps
6. **Exports summary data** - CSV files for further analysis

## Requirements

Install dependencies:

```powershell
pip install matplotlib seaborn numpy pandas
```

Or use the requirements file:

```powershell
pip install -r requirements_enrichment.txt
```

## Usage

Simply run the script from the `scripts` directory:

```powershell
cd scripts
python analyze_enriched_data.py
```

The script will:
- Analyze the first 110 rows of `HatefulData_enriched.csv`
- Look at columns 12-24 (the label columns)
- Generate visualizations and statistics
- Save output files to `analysis/` folder

## Output Files

All files are saved in the `analysis/` folder:

### 1. `analysis/label_analysis.png`
Main visualization with 4 plots:
- Label frequency bar chart
- Co-occurrence heatmap
- Labels per text distribution
- Hate vs non-hate comparison

### 2. `label_analysis_detailed.png`
Detailed analysis with 4 additional plots:
- Top 5 labels pie chart
- Label correlation matrix
- Cumulative distribution
- Most common label combinations

### 3. `analysis/label_summary.csv`
Summary statistics for each label:
- Count and percentage
- Occurrences in hate vs non-hate texts

### 4. `analysis/label_cooccurrence_matrix.csv`
Full co-occurrence matrix showing which labels appear together

## Example Output

```
======================================================================
1. INDIVIDUAL LABEL FREQUENCIES
======================================================================
incitement_of_violence                  :  15 ( 13.6%)
praising_violence                       :   8 (  7.3%)
praising_extremist_acts                 :   5 (  4.5%)
recruitment_for_extremism               :   3 (  2.7%)
racism                                  :  45 ( 40.9%)
...

Total label instances: 234
Average labels per text: 2.13

======================================================================
2. LABEL CO-OCCURRENCES
======================================================================
Top 10 Label Co-occurrences:
 1. racism                  + targeting_ethnic_or_racial_groups: 28 (25.5%)
 2. incitement_of_violence  + physical_violence                : 12 (10.9%)
 3. sexual_harassment       + personal_attacks                 :  9 ( 8.2%)
...
```

## Customization

### Analyze Different Rows

Edit line 14 in the script:

```python
df_subset = df.head(110)  # Change 110 to any number
```

### Analyze Different Columns

Edit line 20 in the script:

```python
label_columns = df.columns[11:24].tolist()  # Change indices as needed
```

### Change Output File Names

Edit the output folder name at the top of the script:

```python
output_folder = "analysis"  # Change to any folder name
```

Or edit individual file paths:

```python
output_path = os.path.join(output_folder, 'my_analysis.png')
```

## Troubleshooting

**File not found error**:
- Make sure `HatefulData_enriched.csv` exists in the parent directory
- Run the script from the `scripts` folder

**Column index error**:
- Check that your CSV has the expected columns
- Adjust the column indices if your structure is different

**Import errors**:
- Install missing packages: `pip install matplotlib seaborn numpy`

## Understanding the Analysis

### Co-occurrence Analysis
Shows which labels tend to appear together. High co-occurrence suggests related concepts.

### Correlation Matrix
Shows statistical correlation between labels (-1 to 1):
- **Positive correlation**: Labels often appear together
- **Negative correlation**: Labels rarely appear together
- **Zero correlation**: No relationship

### Sparsity
Percentage of label slots that are empty (0). High sparsity means most texts have few labels.

### Label Combinations
Most common sets of multiple labels assigned to the same text.

## Tips

1. **Start with the visualizations** - They give you a quick overview
2. **Check co-occurrences** - Understand which hate types are related
3. **Compare hate vs non-hate** - See if enrichment makes sense
4. **Look at combinations** - Find common patterns in hate speech

## Next Steps

After running the analysis:

1. Review the visualizations to understand your data
2. Check if label distributions make sense
3. Look for unexpected co-occurrences
4. Use insights to refine your labeling strategy
5. Consider retraining models on the enriched labels
