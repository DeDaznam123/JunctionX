import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Create output directory for charts
output_dir = Path(__file__).parent / 'charts'
output_dir.mkdir(exist_ok=True)

# Load the enriched datasets
hateful_data = pd.read_csv('../../data/enriched_data/HatefulData_enriched.csv')
annotations_data = pd.read_csv('../../data/enriched_data/annotations_data_final.csv')

print(f"HatefulData_enriched shape: {hateful_data.shape}")
print(f"Annotations_data shape: {annotations_data.shape}")

# Define category columns
hate_categories = [
    'Incitement of Violence', 'Praising violence', 'Praising extremist acts',
    'Targeting ethnic or racial groups', 'Ideologically motivated threats',
    'Anti-democratic rhetoric', 'Personal Attacks', 'Sexual harassment',
    'Physical violence', 'Psychological attacks'
]

# ==================== HATEFUL DATA ANALYSIS ====================

# 1. Distribution of Hate vs Non-Hate Speech
plt.figure(figsize=(10, 6))
label_counts = hateful_data['label'].value_counts()
colors = ['#e74c3c', '#2ecc71']
plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('Distribution of Hate vs Non-Hate Speech', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / '1_hate_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 1_hate_distribution.png")
plt.close()

# 2. Hate Category Frequency Analysis
plt.figure(figsize=(14, 8))
category_sums = hateful_data[hate_categories].sum().sort_values(ascending=True)
colors_bar = plt.cm.RdYlBu_r(np.linspace(0.3, 0.9, len(category_sums)))
category_sums.plot(kind='barh', color=colors_bar)
plt.xlabel('Frequency', fontsize=12, fontweight='bold')
plt.title('Frequency of Hate Speech Categories', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / '2_category_frequency.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 2_category_frequency.png")
plt.close()


# 3. Category Frequency in Annotations Data
plt.figure(figsize=(14, 8))
ann_category_sums = annotations_data[hate_categories].sum().sort_values(ascending=True)
colors_bar = plt.cm.RdYlBu_r(np.linspace(0.3, 0.9, len(ann_category_sums)))
ann_category_sums.plot(kind='barh', color=colors_bar)
plt.xlabel('Frequency', fontsize=12, fontweight='bold')
plt.title('Annotations Data: Frequency of Hate Speech Categories', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / '3_annotations_category_frequency.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 3_annotations_category_frequency.png")
plt.close()


# 4. Correlation Heatmap of Hate Categories
plt.figure(figsize=(12, 10))
correlation_matrix = hateful_data[hate_categories].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Between Hate Speech Categories', fontsize=16, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(output_dir / '4_category_correlation.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 4_category_correlation.png")
plt.close()

# 5. Co-occurrence Matrix of Hate Categories
plt.figure(figsize=(12, 10))
co_occurrence = hateful_data[hate_categories].T.dot(hateful_data[hate_categories])
sns.heatmap(co_occurrence, annot=True, fmt='g', cmap='YlOrRd', square=True,
            linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Co-occurrence of Hate Speech Categories', fontsize=16, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(output_dir / '5_category_cooccurrence.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 5_category_cooccurrence.png")
plt.close()

# 6. Multiple Categories per Text
hateful_data['category_count'] = hateful_data[hate_categories].sum(axis=1)
plt.figure(figsize=(10, 6))
category_count_dist = hateful_data['category_count'].value_counts().sort_index()
plt.bar(category_count_dist.index, category_count_dist.values, color='#9b59b6', edgecolor='black')
plt.xlabel('Number of Hate Categories per Text', fontsize=12, fontweight='bold')
plt.ylabel('Frequency', fontsize=12, fontweight='bold')
plt.title('Distribution of Multiple Hate Categories per Text', fontsize=16, fontweight='bold')
plt.xticks(range(int(category_count_dist.index.max()) + 1))
plt.tight_layout()
plt.savefig(output_dir / '6_multiple_categories.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 6_multiple_categories.png")
plt.close()


# ==================== SUMMARY STATISTICS ====================

print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

print("\n--- HatefulData_enriched.csv ---")
print(f"Total samples: {len(hateful_data)}")
print(f"Hate speech: {len(hateful_data[hateful_data['label'] == 'hate'])} ({len(hateful_data[hateful_data['label'] == 'hate'])/len(hateful_data)*100:.1f}%)")
print(f"Non-hate speech: {len(hateful_data[hateful_data['label'] == 'nothate'])} ({len(hateful_data[hateful_data['label'] == 'nothate'])/len(hateful_data)*100:.1f}%)")
model_wrong_count = hateful_data['model_wrong'].fillna(False).astype(bool).sum()
model_accuracy = (len(hateful_data) - model_wrong_count) / len(hateful_data) * 100
print(f"Model accuracy: {model_accuracy:.1f}%")
print(f"Average categories per text: {hateful_data['category_count'].mean():.2f}")
print(f"Most common category: {category_sums.idxmax()} ({category_sums.max()} occurrences)")

print("\n--- annotations_data_final.csv ---")
print(f"Total samples: {len(annotations_data)}")
print(f"Hate speech: {len(annotations_data[annotations_data['label'] == 'hate'])} ({len(annotations_data[annotations_data['label'] == 'hate'])/len(annotations_data)*100:.1f}%)")
print(f"Non-hate speech: {len(annotations_data[annotations_data['label'] == 'nothate'])} ({len(annotations_data[annotations_data['label'] == 'nothate'])/len(annotations_data)*100:.1f}%)")
print(f"Unique users: {annotations_data['user_id'].nunique()}")
print(f"Unique subforums: {annotations_data['subforum_id'].nunique()}")
print(f"Most common category: {ann_category_sums.idxmax()} ({ann_category_sums.max()} occurrences)")

print("\n" + "="*60)
print(f"All charts saved to: {output_dir.absolute()}")
print("="*60)
