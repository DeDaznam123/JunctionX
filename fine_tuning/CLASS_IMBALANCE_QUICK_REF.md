# Quick Reference: Class Imbalance Handling

## What Changed?

✅ **Custom class weights** for severely underrepresented labels
✅ **Focal loss** to focus on hard examples  
✅ **Hybrid approach** preventing overfitting while ensuring learning

## Before Training: Analyze Your Data

```bash
cd fine_tuning
python analyze_class_distribution.py --data_dir ./data
```

This will show:
- Sample counts per label
- Imbalance ratios
- Applied weights
- Visualization (saved as PNG)

## Training with New Features

### Basic Training (recommended)
```bash
python train_model.py --bf16
```

### Custom Focal Loss
```bash
# More aggressive (focus heavily on hard examples)
python train_model.py --focal_gamma 3.0 --bf16

# Less aggressive
python train_model.py --focal_gamma 1.5 --bf16
```

## What to Monitor

### During Training:
```
secondary_f1_macro    # Shows if rare labels are learning
secondary_recall      # Must be > 0 for rare labels
secondary_precision   # Should stay > 0.6
```

### Good Training:
- ✅ `secondary_f1_macro` increases each epoch
- ✅ Rare labels show non-zero recall
- ✅ Precision stays above 0.6

### Bad Training:
- ❌ Rare labels stuck at 0 recall → Increase weights
- ❌ Precision < 0.5 → Decrease weights  
- ❌ Training loss oscillates → Reduce learning rate

## Tuning Weights

Edit `fine_tuning/train_model.py`, line ~400:

```python
custom_weights = {
    'anti-democratic rhetoric': 10.0,      # Adjust this
    'praising extremist acts': 10.0,       # Adjust this
    'praising violence': 5.0,              # Adjust this
    'ideologically motivated threats': 5.0, # Adjust this
    'sexual harassment': 5.0,              # Adjust this
}
```

**Guidelines:**
- Rare labels (100:1): Start with 10x, can go up to 15x
- Medium labels (10:1): Start with 5x, can go up to 8x
- Never exceed 20x (causes instability)

## File Changes Summary

### Modified Files:
1. `train_model.py`:
   - `MultiLabelRobertaWithConstraints.__init__`: Added class_weights, focal_gamma
   - `compute_constrained_loss`: Implemented weighted focal loss
   - `calculate_class_weights`: New method to compute weights
   - `create_model`: Now accepts weights and focal_gamma
   - `train`: Calculates and uses class weights

### New Files:
1. `CLASS_IMBALANCE_SOLUTION.md`: Full documentation
2. `analyze_class_distribution.py`: Analysis tool
3. `CLASS_IMBALANCE_QUICK_REF.md`: This file

## Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| Rare labels have 0 recall | Increase custom weights (10→15) |
| Model overpredicts rare labels | Decrease custom weights (10→7) |
| Training unstable/oscillating | Reduce learning rate (2e-5→1e-5) |
| Training too slow | Increase batch size (16→32) |
| Out of memory | Decrease batch size or max_length |

## Example Training Commands

### Standard Training:
```bash
python train_model.py \
  --num_epochs 3 \
  --batch_size 16 \
  --learning_rate 2e-5 \
  --focal_gamma 2.0 \
  --bf16
```

### If Rare Labels Not Learning:
```bash
# 1. First, increase focal_gamma
python train_model.py --focal_gamma 3.0 --bf16

# 2. If still not working, edit custom_weights in code
#    and increase from 10.0 to 15.0
```

### If Overpredicting:
```bash
# Reduce focal_gamma
python train_model.py --focal_gamma 1.5 --bf16

# Then edit custom_weights in code and decrease to 7.0
```

## Technical Details

**Loss Function:**
```
L = w_i × (1-p_i)^γ × BCE(y_i, p_i) + 0.5 × constraint_penalty
```

Where:
- `w_i`: Class weight for label i
- `γ`: Focal gamma (default 2.0)
- `p_i`: Predicted probability
- `y_i`: True label

**Effect of Focal Loss (γ=2.0):**
- Easy examples (p > 0.9): Weight reduced to ~1%
- Hard examples (p < 0.1): Weight stays at ~80-100%

This ensures the model spends more time learning difficult cases!
