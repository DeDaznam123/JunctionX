# Class Imbalance Solution for Underrepresented Labels

## Problem
Some secondary labels are drastically underrepresented:
- **Anti-democratic rhetoric**: ~100x less than top label
- **Praising extremist acts**: ~100x less than top label
- **Praising violence**: ~10x less than top label
- **Ideologically motivated threats**: ~10x less than top label
- **Sexual harassment**: ~10x less than top label

## Solution Implemented

We've implemented a **hybrid approach** combining:

1. **Custom Class Weighting**: Different weights based on severity of imbalance
2. **Focal Loss**: Automatically focuses on hard-to-classify examples
3. **Constraint Enforcement**: Maintains primary/secondary label relationships

### 1. Class Weighting Strategy

The model now uses customized weights for each label:

```python
# Severely underrepresented (100:1 ratio)
- Anti-democratic rhetoric: 10x weight
- Praising extremist acts: 10x weight

# Moderately underrepresented (10:1 ratio)
- Praising violence: 5x weight
- Ideologically motivated threats: 5x weight
- Sexual harassment: 5x weight

# Other labels: Automatically capped at 3x
# Primary label: Capped at 2x
```

**Why moderate weights (5-10x) instead of 100x?**
- Prevents training instability
- Avoids overfitting on rare examples
- Reduces risk of overpredicting rare labels
- Combined with focal loss for better results

### 2. Focal Loss Component

Focal loss automatically down-weights easy examples and focuses training on hard cases:

```python
focal_weight = (1 - prob)^gamma  # For positive labels
focal_weight = prob^gamma        # For negative labels
```

With gamma=2.0 (default):
- Easy examples (prob > 0.9): weight ~0.01 (99% reduction)
- Medium examples (prob ~0.5): weight ~0.25
- Hard examples (prob < 0.1): weight ~0.81-1.0

### 3. Combined Loss Function

```
Total Loss = Focal_Loss(BCE_weighted) + 0.5 * Constraint_Penalty
```

Where:
- `BCE_weighted`: Binary cross-entropy with custom class weights
- `Focal_Loss`: Down-weights easy examples
- `Constraint_Penalty`: Enforces primary/secondary label rules

## Usage

### Training with Default Settings

```bash
python train_model.py --bf16
```

### Adjusting Focal Loss Gamma

Higher gamma = more focus on hard examples:

```bash
# More aggressive (focus heavily on hard examples)
python train_model.py --focal_gamma 3.0 --bf16

# Less aggressive (more balanced)
python train_model.py --focal_gamma 1.5 --bf16
```

### Monitoring Performance

The training will log:
- Per-label sample counts
- Calculated weights for each label
- Overall metrics (F1, precision, recall)
- Primary label metrics (hate/nothate)
- Secondary label metrics (only for hate samples)

Watch these metrics during training:
```
secondary_f1_micro    # Overall secondary label performance
secondary_f1_macro    # Average per-label performance (shows rare label learning)
secondary_precision   # Are we overpredicting?
secondary_recall      # Are we catching rare labels?
```

## Expected Outcomes

### Good Signs ✅
- `secondary_f1_macro` increases over epochs
- Rare labels show non-zero recall
- `secondary_precision` stays > 0.6
- Training loss decreases steadily

### Warning Signs ⚠️
- `secondary_precision` drops below 0.5 (overpredicting)
- Training loss oscillates wildly (weights too high)
- `secondary_recall` stays at 0 for rare labels (weights too low)
- Validation loss increases while training loss decreases (overfitting)

## Tuning Recommendations

### If Rare Labels Have 0 Recall After Training:

1. **Increase weights** in `calculate_class_weights`:
   ```python
   custom_weights = {
       'anti-democratic rhetoric': 15.0,  # Increased from 10
       'praising extremist acts': 15.0,   # Increased from 10
       # ...
   }
   ```

2. **Increase focal gamma**:
   ```bash
   python train_model.py --focal_gamma 3.0 --bf16
   ```

### If Model Overpredicts Rare Labels:

1. **Decrease weights** in `calculate_class_weights`:
   ```python
   custom_weights = {
       'anti-democratic rhetoric': 7.0,   # Decreased from 10
       'praising extremist acts': 7.0,    # Decreased from 10
       # ...
   }
   ```

2. **Adjust threshold** at inference time (see `inference.py`)

### If Training is Unstable:

1. **Reduce learning rate**:
   ```bash
   python train_model.py --learning_rate 1e-5 --bf16
   ```

2. **Reduce focal gamma**:
   ```bash
   python train_model.py --focal_gamma 1.5 --bf16
   ```

## Technical Details

### Code Changes

1. **`MultiLabelRobertaWithConstraints.__init__`**:
   - Added `class_weights` parameter
   - Added `focal_gamma` parameter
   - Registers weights as buffer for GPU transfer

2. **`MultiLabelRobertaWithConstraints.compute_constrained_loss`**:
   - Uses `BCEWithLogitsLoss(pos_weight=class_weights)`
   - Implements focal loss weighting
   - Combines with constraint penalty

3. **`HateSpeechTrainer.calculate_class_weights`**:
   - Calculates class weights from training data
   - Applies custom weights for severely underrepresented labels
   - Caps automatic weights to prevent instability
   - Logs detailed weight information

4. **`HateSpeechTrainer.train`**:
   - Calculates class weights before model creation
   - Passes weights to model initialization
   - Passes focal_gamma to model

### Mathematical Foundation

**Weighted BCE Loss**:
```
L_BCE = -w * [y * log(p) + (1-y) * log(1-p)]
```

**Focal Loss**:
```
L_Focal = -(1-p)^γ * y * log(p) - p^γ * (1-y) * log(1-p)
```

**Combined with Weights**:
```
L = w * (1-p)^γ * BCE(y, p)
```

This gives rare classes both:
- **More importance** (via weight `w`)
- **Focus on hard examples** (via focal term `(1-p)^γ`)

## References

- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) - Lin et al., 2017
- [Class-Balanced Loss Based on Effective Number of Samples](https://arxiv.org/abs/1901.05555) - Cui et al., 2019
