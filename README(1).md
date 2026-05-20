# ✈ Airline Flight Delay Classifier
### KD34403 — Machine Learning for Data Science | Group 11 | Universiti Malaysia Sabah

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Project Overview

This project builds a **Random Forest binary classifier** to predict whether a US domestic flight will be delayed (>15% of flights on a route delayed in a given month). We use the [Airline Delay Dataset](https://www.kaggle.com/datasets/sriharshaeedala/airline-delay) from Kaggle.

| Item | Detail |
|---|---|
| **Dataset** | Airline Delay — `airline_delay_cleaned.csv` |
| **Task** | Binary Classification (Delayed / Not Delayed) |
| **Model** | RandomForestClassifier |
| **Test Accuracy** | **74.24%** |
| **AUC-ROC** | **0.820** |

---

## 🗂 Repository Structure

```
├── milestone1/
│   └── milestone1_data_pipeline.py       # Data loading, cleaning, preprocessing, EDA
├── milestone2/
│   └── milestone2_architecture_logic.py  # Model architecture & justification
├── milestone3/
│   └── milestone3_training_loop.py       # OOB error curve, learning curve, metrics
├── milestone4/
│   └── milestone4_model_optimization.py  # Hyperparameter tuning, overfitting analysis
├── milestone5/
│   └── milestone5_final_evaluation.py    # Test set evaluation, ROC, feature importance
├── outputs/
│   └── (generated CSVs, PNGs, PKL files go here)
├── run_all.py                            # ⚡ Run all milestones in one go
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run (Reproducibility)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place the dataset
Download `airline_delay_cleaned.csv` from [Kaggle](https://www.kaggle.com/datasets/sriharshaeedala/airline-delay) and place it in the **root folder**.

### 3. Run all milestones in one go ⚡ (Recommended)

```bash
python run_all.py
```

This will automatically run all 5 milestones **in order**, stop immediately if any script fails, and print progress for each step.

<details>
<summary>▶ Or run milestones individually</summary>

```bash
python milestone1/milestone1_data_pipeline.py
python milestone2/milestone2_architecture_logic.py
python milestone3/milestone3_training_loop.py
python milestone4/milestone4_model_optimization.py
python milestone5/milestone5_final_evaluation.py
```
</details>

Each script saves its outputs (CSVs, PNG charts, `.pkl` model) to the `outputs/` folder.

---

## 🚀 run_all.py

```python
import subprocess
import sys

milestones = [
    "milestone1/milestone1_data_pipeline.py",
    "milestone2/milestone2_architecture_logic.py",
    "milestone3/milestone3_training_loop.py",
    "milestone4/milestone4_model_optimization.py",
    "milestone5/milestone5_final_evaluation.py",
]

for script in milestones:
    print(f"▶ Running {script}...")
    subprocess.run([sys.executable, script], check=True)

print("✅ All milestones complete!")
```

> Works on **Windows, macOS, and Linux** — no shell needed.

---

## 📊 Results Summary

| Split | Accuracy |
|---|---|
| Train | 74.91% |
| Validation | 75.04% |
| **Test** | **74.24%** |
| AUC-ROC | 0.820 |

### Feature Importance (Gini Impurity)
| Feature | Importance |
|---|---|
| year | 0.3864 |
| month | 0.2247 |
| carrier_encoded | 0.2223 |
| arr_flights | 0.1250 |
| airport_encoded | 0.0415 |

---

## 🎬 Milestone Presentation Videos

| Milestone | Topic | Presenter | Link |
|---|---|---|---|
| M1 | Data Pipeline | Member 1 | *(add YouTube link)* |
| M2 | Architecture Logic | Member 2 | *(add YouTube link)* |
| M3 | Training Loop | Member 3 | *(add YouTube link)* |
| M4 | Model Optimization | Member 4 | *(add YouTube link)* |
| M5 | Final Evaluation | Member 5 | *(add YouTube link)* |

---

## 👥 Group Members

| Name | Student ID | Milestone |
|---|---|---|
| Member 1 | *(ID)* | M1 — Data Pipeline |
| Member 2 | *(ID)* | M2 — Architecture Logic |
| Member 3 | *(ID)* | M3 — Training Loop |
| Member 4 | *(ID)* | M4 — Model Optimization |
| Member 5 | *(ID)* | M5 — Final Evaluation |

---

## 📦 Dependencies

See `requirements.txt`. Key packages:
- `pandas`, `numpy` — data manipulation
- `scikit-learn` — model training & evaluation
- `matplotlib`, `seaborn` — visualisation
- `joblib` — model persistence
