# ============================================================
# MILESTONE 4 — MODEL OPTIMIZATION
# KD34403 | Group 11 | Airline Flight Delay Classifier
# ============================================================
# Tackles overfitting via:
#   - max_depth sweep
#   - min_samples_split tuning
#   - min_samples_leaf tuning
#   - max_features tuning
# Outputs (saved to ../outputs/):
#   m4_depth_sweep.png
#   m4_param_comparison.png
#   flight_delay_rf_m4.pkl
#   milestone4_summary.csv
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score

OUT  = os.path.join(os.path.dirname(__file__), '..', 'outputs')
SEED = 42
np.random.seed(SEED)
plt.style.use('seaborn-v0_8-whitegrid')

# ── Load data ─────────────────────────────────────────────────
train_bal = pd.read_csv(os.path.join(OUT, 'train_preprocessed_balanced.csv'))
test_df   = pd.read_csv(os.path.join(OUT, 'test_preprocessed_scaled.csv'))
m3_summary = pd.read_csv(os.path.join(OUT, 'milestone3_summary.csv'))
best_n    = int(m3_summary['Best_n_estimators'].iloc[0])

final_features = ['year', 'month', 'carrier_encoded', 'airport_encoded', 'arr_flights']

X_train = train_bal[final_features]
y_train = train_bal['target']
X_test  = test_df[final_features]
y_test  = test_df['target']

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train
)

# ── 1. Max Depth Sweep ────────────────────────────────────────
print("Sweeping max_depth (5 → 30)...")
depths      = [5, 8, 10, 12, 15, 20, 25, 30, None]
depth_train = []
depth_val   = []

for d in depths:
    rf = RandomForestClassifier(
        n_estimators=best_n, max_depth=d,
        class_weight='balanced', random_state=SEED
    )
    rf.fit(X_tr, y_tr)
    depth_train.append(rf.score(X_tr, y_tr))
    depth_val.append(rf.score(X_val, y_val))
    label = str(d) if d else 'None'
    print(f"  depth={label:<5}  train={depth_train[-1]:.4f}  val={depth_val[-1]:.4f}")

# Plot
depth_labels = [str(d) if d else 'None' for d in depths]
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(depth_labels, depth_train, 'o-', color='#2196F3', linewidth=2, label='Train Accuracy')
ax.plot(depth_labels, depth_val,   's--', color='#F44336', linewidth=2, label='Validation Accuracy')
ax.set_title('Milestone 4 — max_depth Sweep\n(Overfitting Analysis)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('max_depth', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.legend(fontsize=11)
ax.set_ylim([0.68, 1.02])
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'm4_depth_sweep.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: m4_depth_sweep.png")

# Best depth from validation score
best_depth = depths[np.argmax(depth_val)]
print(f"\n✅ Best max_depth: {best_depth}  (val acc = {max(depth_val):.4f})")

# ── 2. Additional Param Comparison ───────────────────────────
print("\nComparing additional regularisation parameters...")

configs = {
    'Baseline\n(M3)'            : {'max_depth': 10, 'min_samples_split': 2,  'min_samples_leaf': 1,  'max_features': 'sqrt'},
    'More\nLeaf Samples'        : {'max_depth': 10, 'min_samples_split': 10, 'min_samples_leaf': 5,  'max_features': 'sqrt'},
    'Shallower\nTrees'          : {'max_depth':  8, 'min_samples_split': 2,  'min_samples_leaf': 1,  'max_features': 'sqrt'},
    'Fewer\nFeatures/Split'     : {'max_depth': 10, 'min_samples_split': 2,  'min_samples_leaf': 1,  'max_features': 'log2'},
    'Optimised\n(M4 Best)'      : {'max_depth': best_depth, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt'},
}
config_train, config_val = [], []

for name, params in configs.items():
    rf = RandomForestClassifier(
        n_estimators=best_n, class_weight='balanced',
        random_state=SEED, **params
    )
    rf.fit(X_tr, y_tr)
    config_train.append(rf.score(X_tr, y_tr))
    config_val.append(rf.score(X_val, y_val))
    print(f"  {name.replace(chr(10),' '):<25}  train={config_train[-1]:.4f}  val={config_val[-1]:.4f}")

x      = np.arange(len(configs))
width  = 0.35
fig, ax = plt.subplots(figsize=(11, 6))
bars1 = ax.bar(x - width/2, config_train, width, label='Train', color='#2196F3', alpha=0.85)
bars2 = ax.bar(x + width/2, config_val,   width, label='Validation', color='#F44336', alpha=0.85)
ax.bar_label(bars1, fmt='%.3f', padding=3, fontsize=8)
ax.bar_label(bars2, fmt='%.3f', padding=3, fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(list(configs.keys()), fontsize=9)
ax.set_ylim([0.68, 1.05])
ax.set_title('Milestone 4 — Parameter Comparison\n(Regularisation Strategies)',
             fontsize=13, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=12)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'm4_param_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: m4_param_comparison.png")

# ── 3. Train & Save Optimised Model ──────────────────────────
optimised_rf = RandomForestClassifier(
    n_estimators     = best_n,
    max_depth        = best_depth,
    min_samples_split= 5,
    min_samples_leaf = 2,
    max_features     = 'sqrt',
    class_weight     = 'balanced',
    random_state     = SEED
)
optimised_rf.fit(X_train, y_train)
test_acc = optimised_rf.score(X_test, y_test)
test_auc = roc_auc_score(y_test, optimised_rf.predict_proba(X_test)[:, 1])

print(f"\n{'='*45}")
print(f"  MILESTONE 4 — OPTIMISED MODEL RESULTS")
print(f"{'='*45}")
print(f"  Test Accuracy : {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"  Test AUC-ROC  : {test_auc:.4f}")
print(f"{'='*45}")

joblib.dump(optimised_rf, os.path.join(OUT, 'flight_delay_rf_m4.pkl'))

summary = pd.DataFrame([{
    'n_estimators': best_n, 'max_depth': best_depth,
    'min_samples_split': 5, 'min_samples_leaf': 2,
    'Test_Accuracy': round(test_acc, 4), 'Test_AUC': round(test_auc, 4)
}])
summary.to_csv(os.path.join(OUT, 'milestone4_summary.csv'), index=False)

print("\n✅ Milestone 4 complete! Run milestone5 next.")
