# ============================================================
# MILESTONE 3 — THE TRAINING LOOP
# KD34403 | Group 11 | Airline Flight Delay Classifier
# ============================================================
# Outputs (saved to ../outputs/):
#   m3_oob_error_curve.png
#   m3_learning_curve.png
#   m3_confusion_roc.png
#   m3_feature_importance.png
#   flight_delay_rf_m3.pkl
#   milestone3_summary.csv
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)

OUT  = os.path.join(os.path.dirname(__file__), '..', 'outputs')
SEED = 42
np.random.seed(SEED)
plt.style.use('seaborn-v0_8-whitegrid')

# ── Load data from Milestone 1 ────────────────────────────────
train_bal = pd.read_csv(os.path.join(OUT, 'train_preprocessed_balanced.csv'))
test_df   = pd.read_csv(os.path.join(OUT, 'test_preprocessed_scaled.csv'))

final_features = ['year', 'month', 'carrier_encoded', 'airport_encoded', 'arr_flights']

X_train_bal = train_bal[final_features]
y_train_bal = train_bal['target']
X_test      = test_df[final_features]
y_test      = test_df['target']

# Internal validation split (80/20 of balanced train)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_bal, y_train_bal, test_size=0.2,
    random_state=SEED, stratify=y_train_bal
)
print(f"Train (balanced) : {X_tr.shape[0]:,}  |  Validation : {X_val.shape[0]:,}  |  Test : {X_test.shape[0]:,}")

# ── CELL 2: OOB Error Curve (Training Loop Proxy) ─────────────
print("\nRunning OOB error sweep (10–200 trees)...")
oob_errors    = []
n_trees_range = list(range(10, 210, 10))

rf_oob = RandomForestClassifier(
    n_estimators=10, max_depth=10, oob_score=True,
    warm_start=True, class_weight='balanced', random_state=SEED
)
for n in n_trees_range:
    rf_oob.set_params(n_estimators=n)
    rf_oob.fit(X_tr, y_tr)
    err = 1 - rf_oob.oob_score_
    oob_errors.append(err)
    if n % 50 == 0 or n == 10:
        print(f"  n={n:>3d}  OOB Error={err:.4f}  OOB Acc={rf_oob.oob_score_:.4f}")

best_n = n_trees_range[np.argmin(oob_errors)]
print(f"\n✅ Best n_estimators: {best_n}  (OOB error = {min(oob_errors):.4f})")

# ── CELL 3: Plot OOB Error Curve ──────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(n_trees_range, oob_errors, marker='o', color='#2196F3',
        linewidth=2, markersize=5, label='OOB Error (training proxy)')
ax.axvline(x=best_n, color='green', linestyle='--', linewidth=1.5,
           label=f'Best: {best_n} trees (OOB error = {min(oob_errors):.4f})')
ax.fill_between(n_trees_range, oob_errors, alpha=0.1, color='#2196F3')
ax.set_title('Milestone 3 — OOB Error vs. Number of Trees\n(Training Loop Progress)',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Number of Trees (n_estimators)', fontsize=12)
ax.set_ylabel('OOB Error Rate', fontsize=12)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'm3_oob_error_curve.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: m3_oob_error_curve.png")

# ── CELL 4: Learning Curve ────────────────────────────────────
print("\nGenerating learning curve (this may take a minute)...")
final_rf = RandomForestClassifier(
    n_estimators=best_n, max_depth=10,
    class_weight='balanced', random_state=SEED
)
train_sizes, train_scores, val_scores = learning_curve(
    estimator=final_rf, X=X_train_bal, y=y_train_bal,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring='accuracy', n_jobs=-1
)
train_mean = train_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
val_mean   = val_scores.mean(axis=1)
val_std    = val_scores.std(axis=1)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(train_sizes, train_mean, 'o-', color='#2196F3', linewidth=2, label='Training Accuracy')
ax.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, alpha=0.15, color='#2196F3')
ax.plot(train_sizes, val_mean, 's--', color='#F44336', linewidth=2, label='Validation Accuracy (CV)')
ax.fill_between(train_sizes, val_mean-val_std, val_mean+val_std, alpha=0.15, color='#F44336')
ax.set_title('Milestone 3 — Learning Curve\n(Train vs Validation Accuracy)',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Training Set Size', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.legend(fontsize=11)
ax.set_ylim([0.5, 1.05])
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'm3_learning_curve.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: m3_learning_curve.png")

# ── CELL 5: Train Final Model & Metrics ───────────────────────
final_rf.fit(X_train_bal, y_train_bal)
train_acc        = final_rf.score(X_train_bal, y_train_bal)
val_acc          = final_rf.score(X_val, y_val)
test_acc         = final_rf.score(X_test, y_test)
y_pred           = final_rf.predict(X_test)
y_pred_prob      = final_rf.predict_proba(X_test)[:, 1]
test_auc         = roc_auc_score(y_test, y_pred_prob)

print(f"\n{'='*45}")
print(f"  MILESTONE 3 — RESULTS")
print(f"{'='*45}")
print(f"  Best n_estimators : {best_n}")
print(f"  Train Accuracy    : {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"  Val   Accuracy    : {val_acc:.4f} ({val_acc*100:.2f}%)")
print(f"  Test  Accuracy    : {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"  Test  AUC-ROC     : {test_auc:.4f}")
print(f"{'='*45}")
print(classification_report(y_test, y_pred, target_names=['Not Delayed', 'Delayed']))

# ── CELL 6: Confusion Matrix + ROC Curve ──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Milestone 3 — Evaluation Metrics', fontsize=15, fontweight='bold')

cm   = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Delayed','Delayed'])
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title('Confusion Matrix (Test Set)', fontweight='bold')

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
axes[1].plot(fpr, tpr, color='#2196F3', linewidth=2.5,
             label=f'Random Forest (AUC = {test_auc:.3f})')
axes[1].plot([0,1],[0,1],'k--', linewidth=1, label='Random Classifier')
axes[1].fill_between(fpr, tpr, alpha=0.1, color='#2196F3')
axes[1].set_title('ROC Curve (Test Set)', fontweight='bold')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'm3_confusion_roc.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: m3_confusion_roc.png")

# ── CELL 7: Feature Importance ────────────────────────────────
feat_df = pd.DataFrame({
    'Feature'   : final_features,
    'Importance': final_rf.feature_importances_
}).sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(feat_df['Feature'], feat_df['Importance'],
               color=['#2196F3' if v == feat_df['Importance'].max()
                      else '#90CAF9' for v in feat_df['Importance']])
ax.bar_label(bars, fmt='%.4f', padding=4, fontsize=10)
ax.set_title('Milestone 3 — Feature Importance\n(Random Forest Gini Impurity)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Importance Score')
ax.set_xlim(0, feat_df['Importance'].max() * 1.2)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'm3_feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: m3_feature_importance.png")

# ── CELL 8: Save Model & Summary ─────────────────────────────
joblib.dump(final_rf, os.path.join(OUT, 'flight_delay_rf_m3.pkl'))

summary = pd.DataFrame([{
    'Best_n_estimators': best_n, 'Max_Depth': 10,
    'Train_Accuracy': round(train_acc, 4), 'Val_Accuracy': round(val_acc, 4),
    'Test_Accuracy' : round(test_acc, 4),  'Test_AUC'    : round(test_auc, 4)
}])
summary.to_csv(os.path.join(OUT, 'milestone3_summary.csv'), index=False)

print("\n✅ Milestone 3 complete! Run milestone4 next.")
