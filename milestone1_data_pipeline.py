# ============================================================
# MILESTONE 1 — DATA PIPELINE
# KD34403 | Group 11 | Airline Flight Delay Classifier
# ============================================================
# Steps:
#   1. Load dataset
#   2. Define target variable
#   3. Train/test split (no leakage)
#   4. Label encoding (fit on train only)
#   5. StandardScaler (fit on train only)
#   6. Handle class imbalance (upsample minority)
#   7. EDA visualisations
#   8. Correlation heatmap
# Outputs (saved to ../outputs/):
#   train_preprocessed_unscaled.csv
#   test_preprocessed_unscaled.csv
#   train_preprocessed_scaled.csv
#   test_preprocessed_scaled.csv
#   train_preprocessed_balanced.csv
#   eda_1_carrier_performance.png
#   eda_2_cancellation_trend.png
#   eda_3_delay_reasons_pie.png
#   eda_4_monthly_trend.png
#   eda_5_carrier_month_heatmap.png
#   correlation_heatmap.png
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample

# ── Output directory ──────────────────────────────────────────
OUT = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUT, exist_ok=True)

SEED = 42
np.random.seed(SEED)

# ── 1. Load Data ──────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv('airline_delay_cleaned.csv')
print(f"  Shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")

# ── 2. Target Definition ──────────────────────────────────────
# A route-month is labelled Delayed (1) if >15% of its flights are delayed
df['delay_ratio'] = df['arr_del15'] / df['arr_flights']
df['target']      = (df['delay_ratio'] > 0.15).astype(int)

print(f"\nTarget distribution:")
print(df['target'].value_counts(normalize=True).rename({0:'Not Delayed', 1:'Delayed'}).round(4))

# ── 3. Train / Test Split ─────────────────────────────────────
# SPLIT FIRST — prevents any data leakage into encoders/scaler
features_raw = ['year', 'month', 'carrier', 'airport', 'arr_flights']
X = df[features_raw]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)
print(f"\nTrain size : {len(X_train):,}")
print(f"Test  size : {len(X_test):,}")

# ── 4. Encoding (fit on train only) ───────────────────────────
le_carrier = LabelEncoder()
le_airport = LabelEncoder()

X_train = X_train.copy()
X_test  = X_test.copy()

X_train['carrier_encoded'] = le_carrier.fit_transform(X_train['carrier'])
X_train['airport_encoded'] = le_airport.fit_transform(X_train['airport'])

# Map test set — unseen labels become -1
X_test['carrier_encoded'] = X_test['carrier'].map(
    lambda s: le_carrier.transform([s])[0] if s in le_carrier.classes_ else -1)
X_test['airport_encoded'] = X_test['airport'].map(
    lambda s: le_airport.transform([s])[0] if s in le_airport.classes_ else -1)

final_features = ['year', 'month', 'carrier_encoded', 'airport_encoded', 'arr_flights']
X_train_final = X_train[final_features]
X_test_final  = X_test[final_features]

# ── 5. Save Unscaled ──────────────────────────────────────────
train_unscaled = pd.concat([X_train_final, y_train.reset_index(drop=True)], axis=1)
test_unscaled  = pd.concat([X_test_final,  y_test.reset_index(drop=True)],  axis=1)
train_unscaled.to_csv(os.path.join(OUT, 'train_preprocessed_unscaled.csv'), index=False)
test_unscaled.to_csv( os.path.join(OUT, 'test_preprocessed_unscaled.csv'),  index=False)

# ── 6. Normalisation (fit on train only) ─────────────────────
scaler         = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_final), columns=final_features)
X_test_scaled  = pd.DataFrame(scaler.transform(X_test_final),       columns=final_features)

train_scaled = pd.concat([X_train_scaled, y_train.reset_index(drop=True)], axis=1)
test_scaled  = pd.concat([X_test_scaled,  y_test.reset_index(drop=True)],  axis=1)
train_scaled.to_csv(os.path.join(OUT, 'train_preprocessed_scaled.csv'), index=False)
test_scaled.to_csv( os.path.join(OUT, 'test_preprocessed_scaled.csv'),  index=False)

# ── 7. Handle Class Imbalance (upsample minority on train) ────
df_majority         = train_scaled[train_scaled.target == 1]
df_minority         = train_scaled[train_scaled.target == 0]
df_minority_up      = resample(df_minority, replace=True,
                               n_samples=len(df_majority), random_state=SEED)
train_balanced      = pd.concat([df_majority, df_minority_up])
train_balanced.to_csv(os.path.join(OUT, 'train_preprocessed_balanced.csv'), index=False)

print("\nFiles saved to outputs/:")
for f in ['train_preprocessed_unscaled.csv', 'test_preprocessed_unscaled.csv',
          'train_preprocessed_scaled.csv',   'test_preprocessed_scaled.csv',
          'train_preprocessed_balanced.csv']:
    print(f"  {f}")

# ── 8. EDA Visualisations ─────────────────────────────────────
print("\nGenerating EDA plots...")
plt.style.use('seaborn-v0_8-whitegrid')

# EDA 1: Average Delay Ratio by Carrier
plt.figure(figsize=(10, 8))
carrier_perf = (df.groupby('carrier_name')['delay_ratio']
                  .mean().sort_values(ascending=False).reset_index())
sns.barplot(data=carrier_perf, x='delay_ratio', y='carrier_name',
            palette='viridis', legend=False)
plt.title('Average Delay Ratio by Carrier', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'eda_1_carrier_performance.png'), dpi=150, bbox_inches='tight')
plt.close()

# EDA 2: Cancellations vs Diversions over Time
cd_data = df.groupby('year')[['arr_cancelled', 'arr_diverted']].sum().reset_index()
plt.figure(figsize=(10, 6))
plt.plot(cd_data['year'], cd_data['arr_cancelled'], marker='o', label='Cancelled', color='red')
plt.plot(cd_data['year'], cd_data['arr_diverted'],  marker='s', label='Diverted',  color='orange')
plt.title('Total Annual Cancellations vs Diversions', fontsize=13, fontweight='bold')
plt.legend(); plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'eda_2_cancellation_trend.png'), dpi=150, bbox_inches='tight')
plt.close()

# EDA 3: Delay Reasons Pie Chart
reasons     = ['carrier_delay','weather_delay','nas_delay','security_delay','late_aircraft_delay']
reason_sums = df[reasons].sum()
plt.figure(figsize=(8, 8))
plt.pie(reason_sums,
        labels=[r.replace('_', ' ').title() for r in reasons],
        autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Delay Minutes by Reason', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'eda_3_delay_reasons_pie.png'), dpi=150, bbox_inches='tight')
plt.close()

# EDA 4: Monthly Delay Trend
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='month', y='delay_ratio', marker='o', color='teal')
plt.title('Monthly Delay Ratio Trend', fontsize=13, fontweight='bold')
plt.xticks(range(1, 13))
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'eda_4_monthly_trend.png'), dpi=150, bbox_inches='tight')
plt.close()

# EDA 5: Carrier × Month Heatmap
top_10 = df['carrier'].value_counts().nlargest(10).index
pivot  = (df[df['carrier'].isin(top_10)]
            .pivot_table(index='carrier', columns='month',
                         values='delay_ratio', aggfunc='mean'))
plt.figure(figsize=(12, 6))
sns.heatmap(pivot, cmap='YlOrRd', annot=True, fmt=".2f")
plt.title('Top 10 Carrier Delay Intensity by Month', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'eda_5_carrier_month_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()

# ── 9. Correlation Heatmap ────────────────────────────────────
plt.figure(figsize=(12, 10))
corr = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=False, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Airline Delay Dataset', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()

print("EDA plots saved ✅")
print("\n✅ Milestone 1 complete! Run milestone2 next.")
