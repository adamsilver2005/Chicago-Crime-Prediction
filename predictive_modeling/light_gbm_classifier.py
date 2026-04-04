"""
LightGBM: Multi-Class Crime Type Classifier
Predicts the type of crime (e.g. THEFT, BATTERY, ASSAULT...)
from time, location, and contextual features.

Run locally: python 01_train_classifier.py
Run on Colab: paste this file into a Colab notebook with T4 GPU
"""

import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# configuration 
N_ESTIMATORS  = 1000       # max number of trees
LEARNING_RATE = 0.05       # how much each tree corrects the previous
MAX_DEPTH     = 8          # max depth of each tree
NUM_LEAVES    = 63         # controls model complexity
EARLY_STOP    = 50         # stop if val score doesn't improve for 50 rounds
RANDOM_STATE  = 42

# Load data 
# If running locally, use the PySpark output CSV
# If running on Colab, replace this path with your Drive path
# or pull directly from BigQuery (see 02_load_to_local.py)
df = pd.read_csv("../../data/model1_classification.csv")
print(f"Loaded {len(df):,} rows")
print(f"\nCrime type distribution:")
print(df["primary_type"].value_counts())

# ── 2. Encode target label ─────────────────────────────────
# LightGBM needs integer labels starting from 0
le = LabelEncoder()
df["label"] = le.fit_transform(df["primary_type"])
num_classes = df["label"].nunique()
print(f"\nNumber of classes: {num_classes}")
print(f"Classes: {list(le.classes_)}")

# ── 3. Features and target ─────────────────────────────────
feature_cols = [
    "year", "month", "day_of_week", "hour_of_day",
    "district", "arrest_int", "domestic_int",
    "is_rush_hour", "is_weekend", "season"
]

X = df[feature_cols]
y = df["label"]

# ── 4. Train / test split ──────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain: {X_train.shape} | Test: {X_test.shape}")

# ── 5. Handle class imbalance ──────────────────────────────
# The dataset is imbalanced (THEFT = 21%, GAMBLING = 0.17%)
# We compute class weights so rare classes get more attention
# Same approach used in the fraud detection project
class_counts = y_train.value_counts().sort_index()
class_weights = len(y_train) / (num_classes * class_counts)
sample_weights = y_train.map(class_weights).values

print(f"\nClass weights (higher = rarer class gets more attention):")
for i, w in enumerate(class_weights):
    print(f"  {le.classes_[i]}: {w:.3f}")

# ── 6. Build LightGBM datasets ────────────────────────────
# LightGBM uses its own Dataset format for efficiency
train_data = lgb.Dataset(
    X_train, label=y_train,
    weight=sample_weights       # apply class weights here
)
val_data = lgb.Dataset(
    X_test, label=y_test,
    reference=train_data        # must reference train for consistent encoding
)

# ── 7. Model parameters ────────────────────────────────────
params = {
    "objective":        "multiclass",       # multi-class classification
    "num_class":        num_classes,         # number of output classes
    "metric":           "multi_logloss",     # loss function to monitor
    "learning_rate":    LEARNING_RATE,
    "max_depth":        MAX_DEPTH,
    "num_leaves":       NUM_LEAVES,
    "min_child_samples": 20,                 # min samples per leaf (prevents overfitting)
    "feature_fraction": 0.8,                 # use 80% of features per tree (like Random Forest)
    "bagging_fraction": 0.8,                 # use 80% of data per tree
    "bagging_freq":     5,                   # apply bagging every 5 iterations
    "verbose":          -1,                  # suppress training output spam
    "random_state":     RANDOM_STATE,
}

# ── 8. Train ───────────────────────────────────────────────
# LightGBM's train() handles the loop for you
# callbacks replace the deprecated early_stopping_rounds param
print("\nTraining LightGBM model...")
callbacks = [
    lgb.early_stopping(EARLY_STOP, verbose=True),
    lgb.log_evaluation(period=50)   # print metrics every 50 rounds
]

model = lgb.train(
    params,
    train_data,
    num_boost_round=N_ESTIMATORS,
    valid_sets=[val_data],
    callbacks=callbacks
)

print(f"\nBest iteration: {model.best_iteration}")

# ── 9. Predict ─────────────────────────────────────────────
# model.predict() returns probabilities for each class
# argmax gives us the predicted class index
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)

# ── 10. Evaluation ─────────────────────────────────────────
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=le.classes_
))

# ── 11. Feature importance plot ────────────────────────────
# Shows which features the model relied on most
# Compare this to your fraud project feature importances
importance_df = pd.DataFrame({
    "feature":    feature_cols,
    "importance": model.feature_importance(importance_type="gain")
}).sort_values("importance", ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(importance_df["feature"], importance_df["importance"])
plt.title("LightGBM Feature Importance (Gain)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("../../outputs/lgbm_feature_importance.png", dpi=150)
print("Saved feature importance plot to outputs/lgbm_feature_importance.png")

# ── 12. Confusion matrix ───────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # normalize by row

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm_pct,
    annot=True, fmt=".2f",
    xticklabels=le.classes_,
    yticklabels=le.classes_,
    cmap="Blues"
)
plt.title("Confusion Matrix (row-normalized)")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("../../outputs/lgbm_confusion_matrix.png", dpi=150)
print("Saved confusion matrix to outputs/lgbm_confusion_matrix.png")

# ── 13. Save model ─────────────────────────────────────────
model.save_model("../../outputs/lgbm_crime_classifier.txt")
print("Saved model to outputs/lgbm_crime_classifier.txt")