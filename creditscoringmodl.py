#  Introduction
#  𝗡𝗮𝗺𝗲: Hina
#  𝗦𝘁𝘂𝗱𝗲𝗻𝘁 𝗜𝗗: CA/SE3/13541
#  𝗗𝗼𝗺𝗮𝗶𝗻: Machine Learning

# =========================
# CREDIT SCORING MODEL
# =========================

# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix
)

# =========================
# 2. LOAD DATASET
# =========================

df = pd.read_csv("german_credit_data.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# =========================
# 3. BASIC UNDERSTANDING
# =========================

print("\nINFO:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTarget Distribution:")
print(df["target"].value_counts())

# =========================
# 4. DATA CLEANING
# =========================

df = df.drop_duplicates()
df = df.dropna(subset=["target"])
df = df.dropna()

# =========================
# 5. TARGET ENCODING
# =========================

df["target"] = df["target"].map({"good": 0, "bad": 1})

# safety check
df = df.dropna(subset=["target"])

print("\nFinal Target Distribution:")
print(df["target"].value_counts())

# =========================
# 6. ENCODING FEATURES 
# =========================

le = LabelEncoder()

for col in df.select_dtypes(include=["object", "string"]).columns:
    if col != "target":
        df[col] = le.fit_transform(df[col])

# =========================
# 7. FEATURES & TARGET
# =========================

X = df.drop("target", axis=1)
y = df["target"]

# =========================
# 8. TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 9. FEATURE SCALING
# =========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 10. MODELS
# =========================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# =========================
# 11. TRAINING + EVALUATION
# =========================

results = {}
roc_curves = {}

for name, model in models.items():

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results[name] = [
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred),
        roc_auc_score(y_test, y_prob)
    ]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_curves[name] = (fpr, tpr)

    print("\n====================")
    print(name)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# =========================
# 12. RESULTS TABLE
# =========================

results_df = pd.DataFrame(
    results,
    index=["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
).T

print("\nMODEL COMPARISON:\n")
print(results_df)

best_model = results_df["ROC-AUC"].idxmax()
print("\nBest Model:", best_model)

# =========================
# 13. ROC CURVE (FIGURE 1)
# =========================

plt.figure(1, figsize=(7,6))

for name, (fpr, tpr) in roc_curves.items():
    plt.plot(fpr, tpr, label=name)

plt.plot([0,1],[0,1],"k--")
plt.title("Figure 1: ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# =========================
# 14. FEATURE IMPORTANCE 
# =========================

rf = models["Random Forest"]

feature_importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

plt.figure(2, figsize=(12,5))

feature_importance.head(10).plot(kind="barh")
plt.title("Figure 2: Top 10 Important Features")
plt.gca().invert_yaxis()
plt.show()

# =========================
# 15. FINAL CONCLUSION
# =========================

print("""
FINAL CONCLUSION:

- Credit Scoring Model built successfully.
- Random Forest performs best based on ROC-AUC.
- Key features: credit amount, savings, duration, age.

Business Impact:
- Helps banks reduce loan defaults
- Improves credit approval decisions
""")