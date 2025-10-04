"""
Script to analyze class distribution and visualize the impact of class weighting.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import ast
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_analyze(data_dir="./data"):
    """Analyze class distribution in training data."""
    data_dir = Path(data_dir)
    
    # Load training data
    train_df = pd.read_csv(data_dir / 'train.csv')
    
    # Load label info
    label_info_path = data_dir / 'label_info.json'
    if label_info_path.exists():
        with open(label_info_path, 'r') as f:
            label_info = json.load(f)
            label_names = label_info.get('label_names', [f"Label_{i}" for i in range(11)])
    else:
        label_names = [f"Label_{i}" for i in range(11)]
    
    # Parse labels
    def parse_labels(label_str):
        try:
            return ast.literal_eval(label_str)
        except:
            return [0] * 11
    
    labels = train_df['labels'].apply(parse_labels).tolist()
    labels_array = np.array(labels)
    
    # Count positive samples for each label
    pos_counts = labels_array.sum(axis=0)
    total_samples = len(labels_array)
    neg_counts = total_samples - pos_counts
    
    print("="*80)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*80)
    print(f"\nTotal samples: {total_samples}")
    print("\nPer-Label Statistics:")
    print("-"*80)
    
    # Define custom weights (same as in train_model.py)
    custom_weights = {
        'anti-democratic rhetoric': 10.0,
        'praising extremist acts': 10.0,
        'praising violence': 5.0,
        'ideologically motivated threats': 5.0,
        'sexual harassment': 5.0,
    }
    
    results = []
    for i in range(len(label_names)):
        label_name = label_names[i]
        pos_count = pos_counts[i]
        neg_count = neg_counts[i]
        pos_ratio = pos_count / total_samples * 100
        
        if pos_count > 0:
            imbalance_ratio = neg_count / pos_count
            
            # Calculate weight
            if i > 0:  # Secondary labels
                if label_name.lower() in custom_weights:
                    weight = custom_weights[label_name.lower()]
                    weight_type = "Custom"
                else:
                    weight = min(imbalance_ratio, 3.0)
                    weight_type = "Auto (capped)"
            else:  # Primary label
                weight = min(imbalance_ratio, 2.0)
                weight_type = "Primary"
        else:
            imbalance_ratio = float('inf')
            weight = 1.0
            weight_type = "No samples"
        
        results.append({
            'index': i,
            'name': label_name,
            'positive': int(pos_count),
            'negative': int(neg_count),
            'pos_ratio': pos_ratio,
            'imbalance': imbalance_ratio,
            'weight': weight,
            'weight_type': weight_type
        })
        
        print(f"\n{i}. {label_name}")
        print(f"   Positive: {int(pos_count):>6} ({pos_ratio:>5.2f}%)")
        print(f"   Negative: {int(neg_count):>6} ({100-pos_ratio:>5.2f}%)")
        print(f"   Imbalance Ratio: {imbalance_ratio:>6.1f}:1")
        print(f"   Applied Weight: {weight:>5.1f}x ({weight_type})")
    
    print("\n" + "="*80)
    print("SEVERELY UNDERREPRESENTED LABELS (custom weights applied):")
    print("="*80)
    
    for r in results:
        if r['weight_type'] == "Custom":
            print(f"\n{r['name']}:")
            print(f"  - Only {r['positive']} samples ({r['pos_ratio']:.3f}%)")
            print(f"  - Imbalance: {r['imbalance']:.0f}:1")
            print(f"  - Applied weight: {r['weight']:.1f}x (prevents overfitting while ensuring learning)")
    
    print("\n" + "="*80)
    print("WEIGHT STRATEGY SUMMARY:")
    print("="*80)
    print("""
Strategy for handling extreme imbalance:

1. SEVERELY UNDERREPRESENTED (100:1 ratio):
   - Anti-democratic rhetoric: 10x weight
   - Praising extremist acts: 10x weight
   → Moderate weights to avoid training instability

2. MODERATELY UNDERREPRESENTED (10:1 ratio):
   - Praising violence: 5x weight
   - Ideologically motivated threats: 5x weight
   - Sexual harassment: 5x weight
   → Lower weights since imbalance is less severe

3. OTHER LABELS:
   - Automatically weighted, capped at 3x
   → Prevents any label from dominating training

4. FOCAL LOSS:
   - Gamma=2.0 (default)
   - Automatically focuses on hard examples
   → Works synergistically with class weights

WHY NOT 100x WEIGHTS?
- Would cause training instability
- High risk of overfitting on few examples
- Model would overpredict rare classes
- Combined approach (moderate weights + focal loss) is more robust
    """)
    
    # Create visualization
    create_visualizations(results)
    
    return results

def create_visualizations(results):
    """Create visualizations of class distribution and weights."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Sort by positive count for better visualization
    sorted_results = sorted(results, key=lambda x: x['positive'], reverse=True)
    
    names = [r['name'][:30] for r in sorted_results]  # Truncate long names
    pos_counts = [r['positive'] for r in sorted_results]
    imbalances = [min(r['imbalance'], 100) for r in sorted_results]  # Cap for visualization
    weights = [r['weight'] for r in sorted_results]
    
    # 1. Positive sample counts (log scale)
    ax1 = axes[0, 0]
    bars1 = ax1.barh(names, pos_counts, color='steelblue')
    ax1.set_xlabel('Number of Positive Samples (log scale)')
    ax1.set_title('Class Distribution - Positive Sample Counts')
    ax1.set_xscale('log')
    ax1.grid(axis='x', alpha=0.3)
    
    # Color severely underrepresented labels
    for i, r in enumerate(sorted_results):
        if r['weight_type'] == 'Custom':
            bars1[i].set_color('darkred')
    
    # 2. Imbalance ratios
    ax2 = axes[0, 1]
    bars2 = ax2.barh(names, imbalances, color='coral')
    ax2.set_xlabel('Imbalance Ratio (Negative:Positive)')
    ax2.set_title('Class Imbalance - Capped at 100:1 for visualization')
    ax2.grid(axis='x', alpha=0.3)
    
    # Color severely underrepresented labels
    for i, r in enumerate(sorted_results):
        if r['weight_type'] == 'Custom':
            bars2[i].set_color('darkred')
    
    # 3. Applied weights
    ax3 = axes[1, 0]
    colors = ['darkred' if r['weight_type'] == 'Custom' 
              else 'orange' if r['weight'] >= 3 
              else 'green' for r in sorted_results]
    bars3 = ax3.barh(names, weights, color=colors)
    ax3.set_xlabel('Applied Weight (multiplier)')
    ax3.set_title('Class Weights Applied During Training')
    ax3.grid(axis='x', alpha=0.3)
    
    # 4. Effective weight (weight * focal)
    ax4 = axes[1, 1]
    # Simulate focal loss effect (assuming average difficulty)
    focal_multiplier = 0.5  # Average effect of focal loss
    effective_weights = [w * focal_multiplier for w in weights]
    bars4 = ax4.barh(names, effective_weights, color='purple', alpha=0.6)
    ax4.set_xlabel('Effective Weight (with focal loss)')
    ax4.set_title('Combined Effect: Class Weights × Focal Loss')
    ax4.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('class_distribution_analysis.png', dpi=300, bbox_inches='tight')
    print("\n📊 Visualization saved as: class_distribution_analysis.png")
    
    # Try to show plot
    try:
        plt.show()
    except:
        print("   (Plot display not available in this environment)")
    plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze class distribution and weights")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    
    args = parser.parse_args()
    
    results = load_and_analyze(args.data_dir)
