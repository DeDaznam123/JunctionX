"""
Analysis script for enriched hate speech dataset
Analyzes label frequencies, co-occurrences, and creates visualizations
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from itertools import combinations
from collections import Counter
import os

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Create analysis output folder
output_folder = "analysis"
os.makedirs(output_folder, exist_ok=True)
print(f"Output folder: {output_folder}/\n")

# Load the enriched dataset
print("Loading HatefulData_enriched.csv...")
df = pd.read_csv("HatefulData_enriched.csv")

# Filter to only rows with label="hate"
if 'label' in df.columns:
    df_hate_only = df[df['label'] == 'hate']
    print(f"Total rows in dataset: {len(df)}")
    print(f"Rows with label='hate': {len(df_hate_only)}")
    print(f"Rows with label='nothate': {len(df[df['label'] == 'nothate'])}")
    df = df_hate_only
else:
    print("Warning: 'label' column not found. Analyzing all rows.")

# Analyze first 110 hate-labeled rows
df_subset = df
print(f"\nAnalyzing first {len(df_subset)} rows with label='hate'")
print(f"Total columns: {len(df.columns)}")
print(f"Column names: {df.columns.tolist()}\n")

# Get label columns (columns 12-24, which is index 11-23 in 0-indexed)
label_columns = df.columns[11:24].tolist()
print(f"Label columns being analyzed: {label_columns}\n")

# ============================================================================
# 1. COUNT INDIVIDUAL LABEL FREQUENCIES
# ============================================================================
print("="*70)
print("1. INDIVIDUAL LABEL FREQUENCIES")
print("="*70)

label_counts = {}
for label in label_columns:
    count = df_subset[label].sum()
    percentage = (count / len(df_subset)) * 100
    label_counts[label] = count
    print(f"{label:40s}: {int(count):3d} ({percentage:5.1f}%)")

print(f"\nTotal label instances: {sum(label_counts.values())}")
print(f"Average labels per text: {sum(label_counts.values()) / len(df_subset):.2f}")

# Count rows with no labels
rows_with_no_labels = (df_subset[label_columns].sum(axis=1) == 0).sum()
rows_with_labels = len(df_subset) - rows_with_no_labels
print(f"Rows with at least one label: {rows_with_labels} ({rows_with_labels/len(df_subset)*100:.1f}%)")
print(f"Rows with no labels (all 0s): {rows_with_no_labels} ({rows_with_no_labels/len(df_subset)*100:.1f}%)")

# ============================================================================
# 2. VISUALIZATION: Bar chart of label frequencies
# ============================================================================
print("\n" + "="*70)
print("2. CREATING VISUALIZATIONS")
print("="*70)

# Sort labels by frequency
sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
labels_sorted = [x[0] for x in sorted_labels]
counts_sorted = [x[1] for x in sorted_labels]

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Hate Speech Label Analysis - HATE-LABELED ROWS ONLY (First 110)', fontsize=16, fontweight='bold')

# Plot 1: Bar chart of label frequencies
ax1 = axes[0, 0]
bars = ax1.barh(labels_sorted, counts_sorted, color='steelblue')
ax1.set_xlabel('Count', fontsize=12)
ax1.set_title('Label Frequency Distribution', fontsize=14, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# Add value labels on bars
for i, (bar, count) in enumerate(zip(bars, counts_sorted)):
    percentage = (count / len(df_subset)) * 100
    ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
             f'{int(count)} ({percentage:.1f}%)', 
             va='center', fontsize=9)

# ============================================================================
# 3. CO-OCCURRENCE ANALYSIS
# ============================================================================
print("\nAnalyzing label co-occurrences...")

# Create co-occurrence matrix
co_occurrence_matrix = pd.DataFrame(0, index=label_columns, columns=label_columns)

for idx, row in df_subset.iterrows():
    active_labels = [label for label in label_columns if row[label] == 1]
    
    # Count co-occurrences
    for label1, label2 in combinations(active_labels, 2):
        co_occurrence_matrix.loc[label1, label2] += 1
        co_occurrence_matrix.loc[label2, label1] += 1

print("\nTop 10 Label Co-occurrences:")
print("-" * 70)

# Get top co-occurrences
co_occur_list = []
for i in range(len(label_columns)):
    for j in range(i+1, len(label_columns)):
        count = co_occurrence_matrix.iloc[i, j]
        if count > 0:
            co_occur_list.append((label_columns[i], label_columns[j], count))

co_occur_list.sort(key=lambda x: x[2], reverse=True)

for i, (label1, label2, count) in enumerate(co_occur_list[:10], 1):
    percentage = (count / len(df_subset)) * 100
    print(f"{i:2d}. {label1:35s} + {label2:35s}: {int(count):3d} ({percentage:.1f}%)")

# Plot 2: Heatmap of co-occurrences
ax2 = axes[0, 1]
sns.heatmap(co_occurrence_matrix, annot=True, fmt='g', cmap='YlOrRd', 
            ax=ax2, cbar_kws={'label': 'Co-occurrence Count'},
            square=True, linewidths=0.5)
ax2.set_title('Label Co-occurrence Heatmap', fontsize=14, fontweight='bold')
ax2.set_xlabel('')
ax2.set_ylabel('')
plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=8)
plt.setp(ax2.get_yticklabels(), rotation=0, fontsize=8)

# ============================================================================
# 4. LABELS PER TEXT DISTRIBUTION
# ============================================================================
print("\n" + "="*70)
print("3. LABELS PER TEXT DISTRIBUTION")
print("="*70)

labels_per_text = df_subset[label_columns].sum(axis=1)
label_count_distribution = labels_per_text.value_counts().sort_index()

print("\nNumber of labels per text:")
for num_labels, count in label_count_distribution.items():
    percentage = (count / len(df_subset)) * 100
    print(f"  {int(num_labels)} labels: {int(count):3d} texts ({percentage:5.1f}%)")

# Plot 3: Distribution of labels per text
ax3 = axes[1, 0]
label_count_distribution.plot(kind='bar', ax=ax3, color='coral')
ax3.set_xlabel('Number of Labels per Text', fontsize=12)
ax3.set_ylabel('Count', fontsize=12)
ax3.set_title('Distribution of Labels per Text', fontsize=14, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)

# Add value labels on bars
for container in ax3.containers:
    ax3.bar_label(container, fontsize=10)

# ============================================================================
# 5. LABEL DISTRIBUTION ANALYSIS (HATE TEXTS ONLY)
# ============================================================================
print("\n" + "="*70)
print("4. LABEL DISTRIBUTION ANALYSIS (HATE TEXTS ONLY)")
print("="*70)

# Since we're only analyzing hate texts, show label distribution
if 'label' in df_subset.columns:
    print(f"\nAnalyzing {len(df_subset)} hate-labeled texts")
    print(f"All texts in this analysis have label='hate'")
    
    # Calculate statistics for hate texts
    avg_labels_hate = df_subset[label_columns].sum().sum() / len(df_subset) if len(df_subset) > 0 else 0
    print(f"Average labels per hate text: {avg_labels_hate:.2f}")
    
    # Show label frequencies
    print("\nLabel frequencies in hate texts:")
    label_freq_data = []
    for label in label_columns:
        count = df_subset[label].sum()
        pct = (count / len(df_subset) * 100) if len(df_subset) > 0 else 0
        label_freq_data.append({
            'label': label,
            'count': count,
            'percentage': pct
        })
        print(f"  {label:40s}: {int(count):3d} ({pct:5.1f}%)")
    
    comparison_df = pd.DataFrame(label_freq_data)
    
    # Plot 4: Bar chart of label percentages in hate texts
    ax4 = axes[1, 1]
    x = np.arange(len(label_columns))
    
    bars = ax4.bar(x, comparison_df['percentage'], color='crimson', alpha=0.8)
    
    ax4.set_xlabel('Labels', fontsize=12)
    ax4.set_ylabel('Percentage of Hate Texts (%)', fontsize=12)
    ax4.set_title('Label Distribution in Hate Texts', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([label.replace('_', '\n') for label in label_columns], fontsize=7, rotation=45, ha='right')
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, pct) in enumerate(zip(bars, comparison_df['percentage'])):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
output_path = os.path.join(output_folder, 'label_analysis.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved visualization to '{output_path}'")

# ============================================================================
# 6. DETAILED STATISTICS
# ============================================================================
print("\n" + "="*70)
print("5. DETAILED STATISTICS")
print("="*70)

print(f"\nDataset Statistics:")
print(f"  Total rows analyzed: {len(df_subset)}")
print(f"  Total label columns: {len(label_columns)}")
print(f"  Total possible label assignments: {len(df_subset) * len(label_columns)}")
print(f"  Actual label assignments: {sum(label_counts.values())}")
print(f"  Sparsity: {(1 - sum(label_counts.values()) / (len(df_subset) * len(label_columns))) * 100:.1f}%")

print(f"\nLabel Statistics:")
print(f"  Most common label: {max(label_counts, key=label_counts.get)} ({label_counts[max(label_counts, key=label_counts.get)]} occurrences)")
print(f"  Least common label: {min(label_counts, key=label_counts.get)} ({label_counts[min(label_counts, key=label_counts.get)]} occurrences)")
print(f"  Median label count: {np.median(list(label_counts.values())):.0f}")

# ============================================================================
# 7. CREATE ADDITIONAL DETAILED VISUALIZATIONS
# ============================================================================
print("\n" + "="*70)
print("6. CREATING ADDITIONAL VISUALIZATIONS")
print("="*70)

# Create a second figure for more detailed analysis
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle('Detailed Hate Speech Label Analysis - HATE-LABELED ROWS ONLY', fontsize=16, fontweight='bold')

# Plot 5: Pie chart showing ALL labels + no labels category
ax5 = axes2[0, 0]

# Count rows with no labels (all 0s)
rows_with_no_labels = (df_subset[label_columns].sum(axis=1) == 0).sum()

# Create dictionary with all labels + no labels category
all_labels_counts = label_counts.copy()
all_labels_counts['No Labels'] = rows_with_no_labels

# Sort by count
sorted_all_labels = sorted(all_labels_counts.items(), key=lambda x: x[1], reverse=True)
pie_labels = [x[0] for x in sorted_all_labels]
pie_values = [x[1] for x in sorted_all_labels]

# Create color palette
colors = plt.cm.Set3(range(len(pie_labels)))

# Create pie chart
wedges, texts, autotexts = ax5.pie(pie_values, labels=pie_labels, 
                                     autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100.*sum(pie_values))})',
                                     colors=colors, startangle=90)
ax5.set_title('Label Distribution (All Labels + No Labels)', fontsize=14, fontweight='bold')
plt.setp(autotexts, size=8, weight="bold")
plt.setp(texts, size=8)

# Add legend with better formatting
ax5.legend(wedges, [f'{label}: {count}' for label, count in sorted_all_labels],
          title="Labels",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1),
          fontsize=8)

# Plot 6: Correlation between labels
ax6 = axes2[0, 1]
correlation_matrix = df_subset[label_columns].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            ax=ax6, center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5)
ax6.set_title('Label Correlation Matrix', fontsize=14, fontweight='bold')
plt.setp(ax6.get_xticklabels(), rotation=45, ha='right', fontsize=8)
plt.setp(ax6.get_yticklabels(), rotation=0, fontsize=8)

# Plot 7: Cumulative label distribution
ax7 = axes2[1, 0]
cumsum = np.cumsum(counts_sorted)
ax7.plot(range(1, len(counts_sorted) + 1), cumsum, marker='o', color='purple', linewidth=2, markersize=8)
ax7.fill_between(range(1, len(counts_sorted) + 1), cumsum, alpha=0.3, color='purple')
ax7.set_xlabel('Number of Labels (ranked)', fontsize=12)
ax7.set_ylabel('Cumulative Count', fontsize=12)
ax7.set_title('Cumulative Label Distribution', fontsize=14, fontweight='bold')
ax7.grid(True, alpha=0.3)
ax7.set_xticks(range(1, len(counts_sorted) + 1))

# Plot 8: Label combinations (most common multi-label combinations)
ax8 = axes2[1, 1]
label_combinations = []
for idx, row in df_subset.iterrows():
    active_labels = tuple(sorted([label for label in label_columns if row[label] == 1]))
    if len(active_labels) > 1:  # Only consider multi-label cases
        label_combinations.append(active_labels)

if label_combinations:
    combo_counts = Counter(label_combinations)
    top_combos = combo_counts.most_common(10)
    
    combo_names = [' + '.join([l.replace('_', ' ')[:15] for l in combo[:2]]) + ('...' if len(combo) > 2 else '') 
                   for combo, _ in top_combos]
    combo_values = [count for _, count in top_combos]
    
    ax8.barh(range(len(combo_names)), combo_values, color='teal')
    ax8.set_yticks(range(len(combo_names)))
    ax8.set_yticklabels(combo_names, fontsize=8)
    ax8.set_xlabel('Frequency', fontsize=12)
    ax8.set_title('Top 10 Label Combinations (Multi-label texts)', fontsize=14, fontweight='bold')
    ax8.grid(axis='x', alpha=0.3)
    
    for i, v in enumerate(combo_values):
        ax8.text(v + 0.1, i, str(v), va='center', fontsize=9)
else:
    ax8.text(0.5, 0.5, 'No multi-label combinations found', 
             ha='center', va='center', fontsize=12, transform=ax8.transAxes)
    ax8.axis('off')

plt.tight_layout()
output_path_detailed = os.path.join(output_folder, 'label_analysis_detailed.png')
plt.savefig(output_path_detailed, dpi=300, bbox_inches='tight')
print(f"✓ Saved detailed visualization to '{output_path_detailed}'")

# ============================================================================
# 8. EXPORT SUMMARY TO CSV
# ============================================================================
print("\n" + "="*70)
print("7. EXPORTING SUMMARY DATA")
print("="*70)

# Create summary dataframe
summary_data = []
for label in label_columns:
    count = int(label_counts[label])
    percentage = round((label_counts[label] / len(df_subset)) * 100, 2)
    summary_data.append({
        'label': label,
        'count': count,
        'percentage': percentage
    })

summary_df = pd.DataFrame(summary_data)
summary_csv_path = os.path.join(output_folder, 'label_summary.csv')
summary_df.to_csv(summary_csv_path, index=False)
print(f"✓ Saved summary data to '{summary_csv_path}'")

# Export co-occurrence matrix
cooccurrence_csv_path = os.path.join(output_folder, 'label_cooccurrence_matrix.csv')
co_occurrence_matrix.to_csv(cooccurrence_csv_path)
print(f"✓ Saved co-occurrence matrix to '{cooccurrence_csv_path}'")

print("\n" + "="*70)
print("✓ ANALYSIS COMPLETE!")
print("="*70)
print(f"\nGenerated files in '{output_folder}/' folder:")
print(f"  1. label_analysis.png - Main visualizations")
print(f"  2. label_analysis_detailed.png - Detailed analysis")
print(f"  3. label_summary.csv - Summary statistics")
print(f"  4. label_cooccurrence_matrix.csv - Co-occurrence data")
