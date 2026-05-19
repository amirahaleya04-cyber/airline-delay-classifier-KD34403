# ============================================================
# MILESTONE 5 — FINAL EVALUATION
# KD34403 | Group 11 | Airline Flight Delay Classifier
# ============================================================
# Outputs (saved to ../outputs/):
#   m5_confusion_matrix.png
#   m5_roc_curve.png
#   m5_feature_importance.png
#   m5_precision_recall_f1.png
#   m5_train_val_test_accuracy.png
#   m5_threshold_precision_recall.png
#   m5_error_analysis.png
#   milestone5_summary.csv
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score, accuracy_score,
    precision_recall_curve
)

OUT  = os.path.join(os.path.dirname(__file__), '..', 'outputs')
SEED = 42
np.random.seed(SEED)
plt.style.use('seaborn-v0_8-whitegrid')

TEAL = '#0E7C7B'
NAVY = '#0D1B2A'
GRAY = '#4A5568'

# ── Load data & model ─────────────────────────────────────────
train_bal  = pd.read_csv(os.path.join(OUT, 'train_preprocessed_balanced.csv'))
test_df    = pd.read_csv(os.path.join(OUT, 'test_preprocessed_scaled.csv'))
m3_summary = pd.read_csv(os.path.join(OUT, 'milestone3_summary.csv'))
best_n     = int(m3_summary['Best_n_estimators'].iloc[0])

final_features = ['year', 'month', 'carrier_encoded', 'airport_encoded', 'arr_flights']

X_train = train_bal[final_features]
y_train = train_bal['target']
X_test  = test_df[final_features]
y_test  = test_df['target']

# Load or re-train model
model_path = os.path.join(OUT, 'flight_delay_rf_m3.pkl')
if os.path.exists(model_path):
    final_rf = joblib.load(model_path)
    print(f"Model loaded from {model_path} ✅")
else:
    final_rf = RandomForestClassifier(
        n_estimators=best_n, max_depth=10,
        class_weight='balanced', random_state=SEED
    )
    final_rf.fit(X_train, y_train)
    print("Model re-trained ✅")

# ── Predictions ───────────────────────────────────────────────
y_pred      = final_rf.predict(X_test)
y_pred_prob = final_rf.predict_proba(X_test)[:, 1]

test_acc = accuracy_score(y_test, y_pred)
test_auc = roc_auc_score(y_test, y_pred_prob)
test_pre = precision_score(y_test, y_pred, average='weighted')
test_rec = recall_score(y_test, y_pred, average='weighted')
test_f1  = f1_score(y_test, y_pred, average='weighted')
train_acc = final_rf.score(X_train, y_train)

print(f"\n{'='*50}")
print(f"  MILESTONE 5 — TEST SET RESULTS")
print(f"{'='*50}")
print(f"  Test Accuracy  : {test_acc:.4f}  ({test_acc*100:.2f}%)")
print(f"  Test AUC-ROC   : {test_auc:.4f}")
print(f"  Precision (wt) : {test_pre:.4f}")
print(f"  Recall    (wt) : {test_rec:.4f}")
print(f"  F1-Score  (wt) : {test_f1:.4f}")
print(f"{'='*50}")
print(classification_report(y_test, y_pred,
      target_names=['Not Delayed (0)', 'Delayed (1)']))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# ── VIZ 1: Confusion Matrix ───────────────────────────────────
fig, ax = plt.subplots(figsize=(5.8, 4.8), facecolor='white')
cm_labels = [[f'TN\n{tn:,}', f'FP\n{fp:,}'], [f'FN\n{fn:,}', f'TP\n{tp:,}']]
bg_cols   = [['#D4EDDA','#F8D7DA'],['#F8D7DA','#D4EDDA']]
txt_col   = [['#1A5E32','#8B0000'],['#8B0000','#1A5E32']]
for i in range(2):
    for j in range(2):
        ax.add_patch(plt.Rectangle((j,1-i),1,1,color=bg_cols[i][j],lw=2,ec='white'))
        ax.text(j+0.5,1.5-i,cm_labels[i][j],ha='center',va='center',
                fontsize=16,fontweight='bold',color=txt_col[i][j],linespacing=1.4)
ax.set_xlim(0,2); ax.set_ylim(0,2)
ax.set_xticks([0.5,1.5]); ax.set_xticklabels(['Not Delayed','Delayed'],fontsize=10,color=GRAY)
ax.set_yticks([0.5,1.5]); ax.set_yticklabels(['Delayed','Not Delayed'],fontsize=10,color=GRAY)
ax.set_xlabel('Predicted Label',fontsize=11,color=GRAY)
ax.set_ylabel('True Label',fontsize=11,color=GRAY)
ax.set_title('Milestone 5 — Confusion Matrix (Test Set)',fontsize=12,fontweight='bold',color=NAVY)
for spine in ax.spines.values(): spine.set_visible(False)
ax.tick_params(length=0)
plt.tight_layout()
plt.savefig(os.path.join(OUT,'m5_confusion_matrix.png'),dpi=150,bbox_inches='tight',facecolor='white')
plt.close(); print("Saved: m5_confusion_matrix.png")

# ── VIZ 2: ROC Curve ─────────────────────────────────────────
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
opt_idx = np.argmax(tpr - fpr)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color=TEAL, linewidth=2.8,
        label=f'Random Forest (AUC = {test_auc:.3f})')
ax.plot([0,1],[0,1],'k--',linewidth=1.2,alpha=0.5,label='Random Classifier (AUC = 0.5)')
ax.fill_between(fpr, tpr, alpha=0.08, color=TEAL)
ax.scatter([fpr[opt_idx]],[tpr[opt_idx]],color='red',s=90,zorder=5,
           label=f'Optimal threshold ≈ {thresholds[opt_idx]:.2f}')
ax.set_xlabel('False Positive Rate',fontsize=11); ax.set_ylabel('True Positive Rate',fontsize=11)
ax.set_title('Milestone 5 — ROC Curve (Test Set)',fontsize=12,fontweight='bold')
ax.legend(fontsize=10); ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
plt.tight_layout()
plt.savefig(os.path.join(OUT,'m5_roc_curve.png'),dpi=150,bbox_inches='tight')
plt.close(); print("Saved: m5_roc_curve.png")

# ── VIZ 3: Feature Importance ─────────────────────────────────
feat_df = pd.DataFrame({'Feature':final_features,
                         'Importance':final_rf.feature_importances_}
                       ).sort_values('Importance',ascending=True)
fig, ax = plt.subplots(figsize=(9,5),facecolor='white')
colors  = [NAVY if v==feat_df['Importance'].max() else TEAL for v in feat_df['Importance']]
bars    = ax.barh(feat_df['Feature'],feat_df['Importance'],color=colors,height=0.55,edgecolor='white')
for bar,val in zip(bars,feat_df['Importance']):
    ax.text(val+0.005,bar.get_y()+bar.get_height()/2,f'{val:.4f}',
            va='center',ha='left',fontsize=10,color=NAVY,fontweight='bold')
ax.set_xlabel('Importance Score',fontsize=11); ax.set_xlim(0,feat_df['Importance'].max()*1.22)
ax.set_title('Milestone 5 — Feature Importance\n(Random Forest Gini Impurity)',
             fontsize=12,fontweight='bold',color=NAVY)
plt.tight_layout()
plt.savefig(os.path.join(OUT,'m5_feature_importance.png'),dpi=150,bbox_inches='tight',facecolor='white')
plt.close(); print("Saved: m5_feature_importance.png")

# ── VIZ 4: Precision / Recall / F1 by Class ──────────────────
report  = classification_report(y_test,y_pred,
          target_names=['Not Delayed','Delayed'],output_dict=True)
classes = ['Not Delayed','Delayed']
metrics = ['precision','recall','f1-score']
lbls    = ['Precision','Recall','F1-Score']
cols_m  = [TEAL,'#F4A261',NAVY]
x,width = np.arange(2), 0.25
fig, ax = plt.subplots(figsize=(8,5))
for i,(m,lbl,col) in enumerate(zip(metrics,lbls,cols_m)):
    vals = [report[c][m] for c in classes]
    b    = ax.bar(x+i*width,vals,width,label=lbl,color=col,alpha=0.88)
    ax.bar_label(b,fmt='%.2f',padding=3,fontsize=9)
ax.set_xticks(x+width); ax.set_xticklabels(classes,fontsize=12)
ax.set_ylim(0,1.15); ax.set_ylabel('Score',fontsize=12)
ax.set_title('Milestone 5 — Precision / Recall / F1 by Class',fontsize=12,fontweight='bold')
ax.legend(fontsize=10)
ax.axhline(test_acc,color='gray',linestyle='--',linewidth=1)
ax.text(2.42,test_acc+0.015,f'Accuracy={test_acc:.2f}',fontsize=8,color='gray')
plt.tight_layout()
plt.savefig(os.path.join(OUT,'m5_precision_recall_f1.png'),dpi=150,bbox_inches='tight')
plt.close(); print("Saved: m5_precision_recall_f1.png")

# ── VIZ 5: Train vs Val vs Test ───────────────────────────────
m3 = pd.read_csv(os.path.join(OUT,'milestone3_summary.csv'))
val_acc_m3 = float(m3['Val_Accuracy'].iloc[0])
splits = ['Train\n({:.2f}%)'.format(train_acc*100),
          'Validation\n({:.2f}%)'.format(val_acc_m3*100),
          'Test\n({:.2f}%)'.format(test_acc*100)]
vals   = [train_acc, val_acc_m3, test_acc]
fig, ax = plt.subplots(figsize=(7,5),facecolor='white')
bars = ax.bar(splits,vals,color=[TEAL,TEAL,NAVY],width=0.45,edgecolor='white')
for bar,v in zip(bars,vals):
    ax.text(bar.get_x()+bar.get_width()/2,v+0.001,f'{v:.4f}',
            ha='center',va='bottom',fontsize=11,fontweight='bold',color=NAVY)
ax.set_ylim(0.70,0.78); ax.set_ylabel('Accuracy',fontsize=12)
ax.set_title('Train vs Validation vs Test Accuracy',fontsize=12,fontweight='bold',color=NAVY)
ax.axhline(test_acc,color='red',linestyle='--',linewidth=1,alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT,'m5_train_val_test_accuracy.png'),dpi=150,bbox_inches='tight',facecolor='white')
plt.close(); print("Saved: m5_train_val_test_accuracy.png")

# ── VIZ 6: Threshold vs Precision & Recall ───────────────────
prec_c, rec_c, pr_thresholds = precision_recall_curve(y_test, y_pred_prob)
fig, ax = plt.subplots(figsize=(9,5))
ax.plot(pr_thresholds,prec_c[:-1],color=TEAL,linewidth=2,label='Precision')
ax.plot(pr_thresholds,rec_c[:-1], color='#F4A261',linewidth=2,label='Recall')
ax.axvline(0.50,color='gray', linestyle='--',linewidth=1.2,label='Default (0.50)')
ax.axvline(thresholds[opt_idx],color='red',linestyle='--',linewidth=1.2,
           label=f'Optimal ({thresholds[opt_idx]:.2f})')
ax.set_xlabel('Decision Threshold',fontsize=12); ax.set_ylabel('Score',fontsize=12)
ax.set_title('Milestone 5 — Precision & Recall vs Decision Threshold',fontsize=12,fontweight='bold')
ax.legend(fontsize=10); ax.set_xlim([0,1]); ax.set_ylim([0,1.05])
plt.tight_layout()
plt.savefig(os.path.join(OUT,'m5_threshold_precision_recall.png'),dpi=150,bbox_inches='tight')
plt.close(); print("Saved: m5_threshold_precision_recall.png")

# ── VIZ 7: Error Analysis ─────────────────────────────────────
total  = len(y_test)
fig, axes = plt.subplots(1,2,figsize=(12,5))
sizes   = [tn,tp,fp,fn]
plabels = ['TN — Correct\nNot Delayed','TP — Correct\nDelayed',
           'FP — Wrong\n(Over-predicted)','FN — Missed\n(Under-predicted)']
pcols   = ['#2DC653',TEAL,'#F4A261','#E63946']
axes[0].pie(sizes,labels=plabels,colors=pcols,explode=(0,0,0.05,0.1),
            autopct='%1.1f%%',startangle=140,textprops={'fontsize':9})
axes[0].set_title('Test Set — Prediction Breakdown',fontsize=11,fontweight='bold')
cats  = ['Correct\n(TN+TP)','False Positives\n(FP)','False Negatives\n(FN)']
cnts  = [tn+tp,fp,fn]
bcols = [TEAL,'#F4A261','#E63946']
b     = axes[1].bar(cats,cnts,color=bcols,width=0.5)
axes[1].bar_label(b,labels=[f'{v:,}\n({v/total*100:.1f}%)' for v in cnts],
                  padding=4,fontsize=10,fontweight='bold')
axes[1].set_ylim(0,max(cnts)*1.2); axes[1].set_ylabel('Count',fontsize=12)
axes[1].set_title('Error Analysis — Correct vs Misclassified',fontsize=11,fontweight='bold')
plt.suptitle('Milestone 5 — Error Analysis (Test Set)',fontsize=13,fontweight='bold',y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT,'m5_error_analysis.png'),dpi=150,bbox_inches='tight')
plt.close(); print("Saved: m5_error_analysis.png")

# ── Save Final Summary ────────────────────────────────────────
summary = pd.DataFrame([{
    'n_estimators': best_n, 'max_depth': 10,
    'Test_Accuracy': round(test_acc,4), 'Test_AUC': round(test_auc,4),
    'Precision_wt' : round(test_pre,4), 'Recall_wt': round(test_rec,4),
    'F1_wt'        : round(test_f1,4),
    'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp),
}])
summary.to_csv(os.path.join(OUT,'milestone5_summary.csv'),index=False)
print("\n✅ Milestone 5 complete! All outputs saved to outputs/")
