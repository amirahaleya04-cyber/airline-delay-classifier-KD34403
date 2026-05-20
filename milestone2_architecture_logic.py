# ============================================================
# MILESTONE 2 — ARCHITECTURE LOGIC
# KD34403 | Group 11 | Airline Flight Delay Classifier
# ============================================================
# Justifies why Random Forest was chosen over alternatives
# and defines the full model architecture with parameters.
# Outputs (saved to ../outputs/):
#   architecture_correlation_heatmap.png
#   milestone2_architecture_summary.txt
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

OUT = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUT, exist_ok=True)

# ── Load data from Milestone 1 ────────────────────────────────
df = pd.read_csv('airline_delay_cleaned.csv')
df['delay_ratio'] = df['arr_del15'] / df['arr_flights']
df['target']      = (df['delay_ratio'] > 0.15).astype(int)

features_raw = ['year', 'month', 'carrier', 'airport', 'arr_flights']
X = df[features_raw]

# ── Define Architecture ───────────────────────────────────────
model = RandomForestClassifier(
    n_estimators  = 100,        # Starting point; optimised in M3 via OOB error
    max_depth     = 10,         # Limits tree depth → prevents overfitting on laptops
    random_state  = 42,         # Reproducibility (Rubric 2b)
    class_weight  = 'balanced'  # Compensates for class imbalance (~67% Delayed)
)

# ── Print Architecture Summary ────────────────────────────────
summary_lines = [
    "=" * 50,
    "MILESTONE 2 — MODEL ARCHITECTURE SUMMARY",
    "=" * 50,
    f"Dataset         : Airline Delay (Kaggle)",
    f"Target Variable : target  (1 = Delayed >15%, 0 = Not Delayed)",
    f"Input Features  : {list(X.columns)}",
    "-" * 50,
    f"Model Type      : {type(model).__name__}",
    f"n_estimators    : {model.n_estimators}  (to be tuned via OOB in M3)",
    f"max_depth       : {model.max_depth}",
    f"class_weight    : {model.class_weight}",
    f"random_state    : {model.random_state}",
    "=" * 50,
    "",
    "WHY RANDOM FOREST?",
    "-" * 50,
    "1. Handles mixed feature types (int, float, encoded categorical)",
    "2. Built-in feature importance — interpretable output",
    "3. Robust to outliers — no assumptions about data distribution",
    "4. class_weight='balanced' natively handles the 67/33 split",
    "5. OOB score = free validation without a separate val set",
    "6. Trains in minutes on standard hardware (no GPU required)",
    "=" * 50,
    "STATUS: Architecture logic ready → proceed to Milestone 3.",
]
print("\n".join(summary_lines))

# Save to file
with open(os.path.join(OUT, 'milestone2_architecture_summary.txt'), 'w') as f:
    f.write("\n".join(summary_lines))
print(f"\nSaved: milestone2_architecture_summary.txt ✅")

# ── Feature Correlation Heatmap ───────────────────────────────
cols = ['target', 'arr_flights', 'carrier_ct', 'weather_ct',
        'nas_ct', 'security_ct', 'late_aircraft_ct']
plt.figure(figsize=(10, 8))
sns.heatmap(df[cols].corr(), annot=True, cmap='RdYlGn', center=0, fmt=".2f")
plt.title('Milestone 2 — Feature Correlation Analysis\n(Technical Justification)',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'architecture_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: architecture_correlation_heatmap.png ✅")

print("\n✅ Milestone 2 complete! Run milestone3 next.")
