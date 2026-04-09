import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# ----- EDIT THESE TWO PATHS WHEN YOU RUN IT -----
IN_PATH = "/home/hpc/v121ca/v121ca21/thesis_results/deberta/deberta_contarga_eval_with_labels.csv"
OUT_DIR = "/home/hpc/v121ca/v121ca21/thesis_results/deberta/metrics"
# -----------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(IN_PATH)

# gold = contarga emotion column
y_true = df["emotion"].astype(str)

# predicted (top1) in GoEmotions space
y_pred = df["pred_top1_emotion"].astype(str)

# basic metrics
acc = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average="macro")
weighted_f1 = f1_score(y_true, y_pred, average="weighted")

print("Rows:", len(df))
print("Accuracy:", round(acc, 4))
print("Macro-F1:", round(macro_f1, 4))
print("Weighted-F1:", round(weighted_f1, 4))

# full report
labels_sorted = sorted(y_true.unique())
report = classification_report(y_true, y_pred, labels=labels_sorted, digits=4, zero_division=0)
with open(os.path.join(OUT_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

# confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
cm_df = pd.DataFrame(cm, index=[f"gold_{l}" for l in labels_sorted], columns=[f"pred_{l}" for l in labels_sorted])
cm_df.to_csv(os.path.join(OUT_DIR, "confusion_matrix.csv"))

# one-line summary csv
summary = pd.DataFrame([{
    "file": IN_PATH,
    "rows": len(df),
    "accuracy": acc,
    "macro_f1": macro_f1,
    "weighted_f1": weighted_f1
}])
summary.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)

print("Saved metrics to:", OUT_DIR)

