"""
STEP 2: Train + compare ML models on the face embeddings, tune the best one,
evaluate on the held-out test set and save model.pkl for the attendance system.

Run:  python train_compare.py
Outputs (in results/): model_comparison.csv, model_comparison.png,
                       confusion_matrix.png, classification_report.txt
"""
import os, time
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_validate, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, ConfusionMatrixDisplay,
                             pairwise_distances)

PROB_T, DIST_T = 0.60, 0.55   # same defaults as attendance.py
os.makedirs("results", exist_ok=True)

data = np.load("features.npz", allow_pickle=True)
X_train, y_train_raw, groups = data["X_train"], data["y_train"], data["groups"]
X_test, y_test_raw, X_unknown = data["X_test"], data["y_test"], data["X_unknown"]

le = LabelEncoder().fit(np.concatenate([y_train_raw, y_test_raw]))
y_train, y_test = le.transform(y_train_raw), le.transform(y_test_raw)
classes = le.classes_.tolist()
print("Classes:", classes)
print("Train:", X_train.shape, " Test:", X_test.shape)

# ---------- models ----------
def pipe(clf):
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

models = {
    "KNN": pipe(KNeighborsClassifier(n_neighbors=3)),
    "SVM (linear)": pipe(SVC(kernel="linear", probability=True, random_state=42)),
    "SVM (RBF)": pipe(SVC(kernel="rbf", probability=True, random_state=42)),
    "Logistic Regression": pipe(LogisticRegression(max_iter=2000)),
    "Random Forest": pipe(RandomForestClassifier(n_estimators=200, random_state=42)),
}
grids = {
    "KNN": {"clf__n_neighbors": [1, 3, 5, 7], "clf__weights": ["uniform", "distance"]},
    "SVM (linear)": {"clf__C": [0.1, 1, 10]},
    "SVM (RBF)": {"clf__C": [1, 10, 100], "clf__gamma": ["scale", 0.01, 0.001]},
    "Logistic Regression": {"clf__C": [0.1, 1, 10, 100]},
    "Random Forest": {"clf__n_estimators": [100, 300], "clf__max_depth": [None, 10]},
}

# group-aware CV: augmented copies of one photo always stay in the same fold
min_groups = min(len(set(groups[y_train == c])) for c in np.unique(y_train))
n_splits = max(2, min(5, min_groups))
cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
print(f"Using {n_splits}-fold stratified group cross-validation\n")

# ---------- compare ----------
rows = []
for name, model in models.items():
    cvr = cross_validate(model, X_train, y_train, groups=groups, cv=cv,
                         scoring=["accuracy", "f1_macro"])
    model.fit(X_train, y_train)
    t0 = time.perf_counter()
    pred = model.predict(X_test)
    ms = (time.perf_counter() - t0) / len(X_test) * 1000
    p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="macro", zero_division=0)
    rows.append({
        "Model": name,
        "CV Accuracy (mean)": cvr["test_accuracy"].mean(),
        "CV Accuracy (std)": cvr["test_accuracy"].std(),
        "CV F1 (mean)": cvr["test_f1_macro"].mean(),
        "Test Accuracy": accuracy_score(y_test, pred),
        "Test Precision": p, "Test Recall": r, "Test F1": f1,
        "Predict ms/face": ms,
    })
    print(f"{name:20s} CV acc {rows[-1]['CV Accuracy (mean)']:.3f} "
          f"(+/-{rows[-1]['CV Accuracy (std)']:.3f}) | test acc {rows[-1]['Test Accuracy']:.3f} "
          f"| {ms:.2f} ms/face")

df = pd.DataFrame(rows).sort_values(["CV F1 (mean)", "Test F1", "Predict ms/face"],
                                    ascending=[False, False, True]).reset_index(drop=True)
df.round(4).to_csv("results/model_comparison.csv", index=False)
print("\n", df.round(4).to_string(index=False))

# bar chart
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(df)); w = 0.38
ax.bar(x - w / 2, df["CV Accuracy (mean)"], w, yerr=df["CV Accuracy (std)"], label="CV accuracy")
ax.bar(x + w / 2, df["Test Accuracy"], w, label="Test accuracy")
ax.set_xticks(x); ax.set_xticklabels(df["Model"], rotation=20)
ax.set_ylim(0, 1.05); ax.set_ylabel("Accuracy"); ax.set_title("Model comparison"); ax.legend()
plt.tight_layout(); plt.savefig("results/model_comparison.png", dpi=150); plt.close()

# ---------- tune the best model ----------
best_name = df.loc[0, "Model"]
print(f"\nBest model by CV F1: {best_name}  -> tuning with GridSearchCV ...")
gs = GridSearchCV(models[best_name], grids[best_name], cv=cv, scoring="f1_macro", n_jobs=-1)
gs.fit(X_train, y_train, groups=groups)
best = gs.best_estimator_
print("Best params:", gs.best_params_, "| CV F1:", round(gs.best_score_, 4))

# ---------- final test evaluation (test set used once) ----------
pred = best.predict(X_test)
print("\n=== FINAL TEST RESULT:", best_name, "===")
print("Test accuracy:", round(accuracy_score(y_test, pred), 4))
report = classification_report(y_test, pred, target_names=classes, zero_division=0)
print(report)
open("results/classification_report.txt", "w").write(
    f"Model: {best_name}\nParams: {gs.best_params_}\n\n{report}")

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=classes,
                                        ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Confusion matrix - {best_name}")
plt.tight_layout(); plt.savefig("results/confusion_matrix.png", dpi=150); plt.close()

# ---------- Unknown-rejection check ----------
def accepted(model, X):
    prob = model.predict_proba(X)
    d = pairwise_distances(X, X_train).min(axis=1)
    return (prob.max(axis=1) >= PROB_T) & (d <= DIST_T)

print(f"Known test faces accepted : {accepted(best, X_test).mean() * 100:.1f}%")
if len(X_unknown):
    print(f"Unknown faces rejected    : {(~accepted(best, X_unknown)).mean() * 100:.1f}%  "
          f"({len(X_unknown)} unknown photos)")
else:
    print("(No dataset/unknown_test photos found - skipped Unknown-rejection test)")

# ---------- save final model (re-fit on train + test = all real data) ----------
X_all = np.vstack([X_train, X_test]); y_all = np.concatenate([y_train, y_test])
final = best
final.fit(X_all, y_all)
joblib.dump({"model": final, "classes": classes, "X_ref": X_all, "model_name": best_name},
            "model.pkl")
print("\nSaved model.pkl and results/ folder")
