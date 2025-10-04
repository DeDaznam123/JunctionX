# Implementation Summary: Class Imbalance Solution

## ✅ Implementation Complete!

Your training script now handles severely underrepresented labels using a **hybrid approach** combining:
- Custom class weighting (5-10x for rare labels)
- Focal loss (automatic hard example mining)
- Constraint enforcement (primary/secondary label relationships)

---

## 🎯 Approach for Your Specific Labels

### Severely Underrepresented (100:1 ratio) → 10x weight
- **Anti-democratic rhetoric**
- **Praising extremist acts**

### Moderately Underrepresented (10:1 ratio) → 5x weight  
- **Praising violence**
- **Ideologically motivated threats**
- **Sexual harassment**

### Why Moderate Weights?
- **Prevents training instability** from extreme 100x weights
- **Reduces overfitting risk** on few examples
- **Combined with focal loss** for synergistic effect
- **Better generalization** than pure reweighting

---

## 📋 What Was Changed

### 1. Model Architecture (`MultiLabelRobertaWithConstraints`)
```python
def __init__(self, ..., class_weights=None, focal_gamma=2.0):
    # Now accepts class weights and focal gamma parameter
    self.register_buffer('class_weights', torch.tensor(class_weights))
    self.focal_gamma = focal_gamma
```

### 2. Loss Function (`compute_constrained_loss`)
```python
# Weighted BCE Loss
loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.class_weights, reduction='none')

# Focal Loss Weighting
focal_weight = (1 - probs) ** self.focal_gamma  # For positives
focal_loss = (focal_weight * bce_loss).mean()

# Combined with constraint penalty
total_loss = focal_loss + 0.5 * constraint_penalty
```

### 3. Weight Calculation (`calculate_class_weights`)
```python
custom_weights = {
    'anti-democratic rhetoric': 10.0,
    'praising extremist acts': 10.0,
    'praising violence': 5.0,
    'ideologically motivated threats': 5.0,
    'sexual harassment': 5.0,
}
# Other labels: auto-weighted, capped at 3x
```

### 4. Training Pipeline (`train`)
- Calculates class weights from training data
- Logs detailed weight information
- Passes weights to model initialization

### 5. CLI Arguments
- Added `--focal_gamma` parameter (default 2.0)

---

## 🚀 How to Use

### Step 1: Analyze Your Data (Optional but Recommended)
```bash
cd fine_tuning
python analyze_class_distribution.py --data_dir ./data
```

This shows:
- Sample counts per label
- Imbalance ratios  
- Applied weights
- Visualization saved as PNG

### Step 2: Train with Default Settings
```bash
python train_model.py --bf16
```

The model will automatically:
1. Load training data
2. Calculate class weights
3. Apply custom weights to underrepresented labels
4. Train with weighted focal loss

### Step 3: Monitor Training
Watch these metrics in the logs:
```
secondary_f1_macro    # Overall per-label performance
secondary_recall      # Must be > 0 for rare labels
secondary_precision   # Should stay > 0.6
```

---

## 🔧 Tuning Guide

### If Rare Labels Show 0 Recall:

**Option 1: Increase Focal Gamma**
```bash
python train_model.py --focal_gamma 3.0 --bf16
```

**Option 2: Increase Custom Weights**  
Edit `train_model.py`, line ~405:
```python
custom_weights = {
    'anti-democratic rhetoric': 15.0,  # Increased from 10
    'praising extremist acts': 15.0,   # Increased from 10
    'praising violence': 7.0,          # Increased from 5
    # ...
}
```

### If Model Overpredicts Rare Labels:

**Option 1: Decrease Focal Gamma**
```bash
python train_model.py --focal_gamma 1.5 --bf16
```

**Option 2: Decrease Custom Weights**
```python
custom_weights = {
    'anti-democratic rhetoric': 7.0,   # Decreased from 10
    'praising extremist acts': 7.0,    # Decreased from 10
    # ...
}
```

### If Training is Unstable:

**Reduce Learning Rate**
```bash
python train_model.py --learning_rate 1e-5 --bf16
```

---

## 📊 Expected Results

### Good Training:
✅ `secondary_f1_macro` increases each epoch  
✅ Rare labels achieve recall > 0.1  
✅ Precision stays > 0.6  
✅ Training loss decreases steadily  

### Warning Signs:
⚠️ Rare labels stuck at 0 recall → Increase weights/gamma  
⚠️ Precision < 0.5 → Decrease weights  
⚠️ Loss oscillates → Reduce learning rate  
⚠️ Val loss increases → Possible overfitting  

---

## 📁 New Files Created

1. **`CLASS_IMBALANCE_SOLUTION.md`**
   - Comprehensive documentation
   - Mathematical foundations
   - Tuning recommendations

2. **`analyze_class_distribution.py`**
   - Analyzes your training data
   - Shows class distribution
   - Visualizes applied weights

3. **`CLASS_IMBALANCE_QUICK_REF.md`**
   - Quick reference guide
   - Common commands
   - Troubleshooting table

4. **`IMPLEMENTATION_SUMMARY.md`** (this file)
   - High-level overview
   - What changed and why
   - How to use

---

## 🎓 Technical Background

### Why This Approach?

**Problem:** Simple reweighting with 100x causes:
- Training instability
- Overfitting on few examples
- Overprediction of rare classes

**Solution:** Hybrid approach with moderate weights + focal loss:
- **Moderate weights (5-10x):** Stable training, less overfitting
- **Focal loss:** Automatically focuses on hard examples
- **Synergistic effect:** Better than either alone

### Mathematical Formulation

**Standard BCE Loss:**
```
L_BCE = -[y log(p) + (1-y) log(1-p)]
```

**Weighted BCE Loss:**
```
L_weighted = -w × [y log(p) + (1-y) log(1-p)]
```

**Focal Loss:**
```
L_focal = -(1-p)^γ × y × log(p) - p^γ × (1-y) × log(1-p)
```

**Our Combined Loss:**
```
L = w × (1-p)^γ × BCE(y, p) + λ × constraint_penalty
```

Where:
- `w`: Class weight (5-10x for rare labels)
- `γ`: Focal gamma (2.0 default)
- `λ`: Constraint weight (0.5)

This gives rare classes:
1. **Higher importance** (via `w`)
2. **Focus on hard examples** (via `(1-p)^γ`)
3. **Stable training** (moderate `w`, not 100x)

---

## 🤝 Next Steps

1. **Run analysis:**
   ```bash
   python analyze_class_distribution.py
   ```

2. **Start training:**
   ```bash
   python train_model.py --bf16
   ```

3. **Monitor logs** for per-label metrics

4. **Tune if needed** based on results

5. **Evaluate on test set:**
   ```bash
   python evaluate_model.py --model_path ./outputs/best_model
   ```

---

## 📚 References

- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- [Class-Balanced Loss Based on Effective Number of Samples](https://arxiv.org/abs/1901.05555)

---

## ❓ Questions?

Check the detailed docs:
- `CLASS_IMBALANCE_SOLUTION.md` - Full documentation
- `CLASS_IMBALANCE_QUICK_REF.md` - Quick reference

Good luck with training! 🚀
