# Imports
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedGroupKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    make_scorer,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Carga de datos

ROOT = Path(__file__).parent.parent
FIGURES = ROOT / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

vst = pd.read_csv(r"C:\Users\alexm\Desktop\proyectos\rnaseq-brca-pipeline\rnaseq-brca-pipeline\data\processed\vst_brca.csv", index_col=0)

metadata = pd.read_csv(
    r"C:\Users\alexm\Desktop\proyectos\rnaseq-brca-pipeline\rnaseq-brca-pipeline\data\processed\metadata_brca.csv",
    index_col=0
)

data = vst.join(metadata[["condition"]], how="inner")

# Split personalizado para evitar que pacientes compartidos entre train y test.

patient_ids = data.index.to_series().str.split("-").str[:3].str.join("-")
outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_indices, test_indices = next(
    outer_cv.split(data, data["condition"], groups=patient_ids)
)

train_set = data.iloc[train_indices]
test_set = data.iloc[test_indices]

# Comprobacion de que no falten muestras
print(vst.shape)
print(metadata.shape)
print(data["condition"].value_counts())
print(vst.index.equals(metadata.index))
print("Shared patients:", set(patient_ids.loc[train_set.index]) & set(patient_ids.loc[test_set.index]))

X_train = train_set.drop(columns=["condition"])
y_train = train_set["condition"]

X_test = test_set.drop(columns=["condition"])
y_test = test_set["condition"]

# Scores de evaluacion

scoring = {
    "tumor_recall": make_scorer(recall_score, pos_label="tumor"),
    "normal_recall": make_scorer(recall_score, pos_label="normal"),
    "roc_auc": "roc_auc",
    "balanced_accuracy": "balanced_accuracy",
}

# Baseline model - Logistic Regression

baseline_model = LogisticRegression(max_iter=1000, random_state=42)
inner_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = cross_validate(
    baseline_model,
    X_train,
    y_train,
    cv=inner_cv,
    groups=patient_ids.loc[X_train.index],
    scoring=scoring,
)

baseline_model.fit(X_train, y_train)

results = {}

results["Baseline LogReg"] = {
    "tumor_recall_mean": cv_results["test_tumor_recall"].mean(),
    "tumor_recall_std": cv_results["test_tumor_recall"].std(),
    "normal_recall_mean": cv_results["test_normal_recall"].mean(),
    "normal_recall_std": cv_results["test_normal_recall"].std(),
    "roc_auc_mean": cv_results["test_roc_auc"].mean(),
    "roc_auc_std": cv_results["test_roc_auc"].std(),
    "balanced_accuracy_mean": cv_results["test_balanced_accuracy"].mean(),
    "balanced_accuracy_std": cv_results["test_balanced_accuracy"].std(),
    "model": baseline_model,
}

for metric in scoring:
    values = cv_results[f"test_{metric}"]

    print(
        f"Baseline LogReg {metric}: "
        f"{values.mean():.4f} ± {values.std():.4f}"
    )

# Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

cv_results_rf = cross_validate(
    rf_model,
    X_train,
    y_train,
    cv=inner_cv,
    groups=patient_ids.loc[X_train.index],
    scoring=scoring,
)

results["Random Forest"] = {
    "tumor_recall_mean": cv_results_rf["test_tumor_recall"].mean(),
    "tumor_recall_std": cv_results_rf["test_tumor_recall"].std(),
    "normal_recall_mean": cv_results_rf["test_normal_recall"].mean(),
    "normal_recall_std": cv_results_rf["test_normal_recall"].std(),
    "roc_auc_mean": cv_results_rf["test_roc_auc"].mean(),
    "roc_auc_std": cv_results_rf["test_roc_auc"].std(),
    "balanced_accuracy_mean": cv_results_rf["test_balanced_accuracy"].mean(),
    "balanced_accuracy_std": cv_results_rf["test_balanced_accuracy"].std(),
    "model": rf_model,
}

for metric in scoring:
    values = cv_results_rf[f"test_{metric}"]

    print(
        f"Random Forest {metric}: "
        f"{values.mean():.4f} ± {values.std():.4f}"
    )

rf_model.fit(X_train, y_train)

# Comparación de modelos y curvas ROC

results_df = pd.DataFrame(results).T
groups_train = patient_ids.loc[X_train.index]

models = {
    "Baseline LogReg": baseline_model,
    "Random Forest": rf_model,
}

plt.figure(figsize=(7, 5))

for name, model in models.items():
    probabilities = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=inner_cv,
        groups=groups_train,
        method="predict_proba",
    )

    tumor_column = list(model.classes_).index("tumor")
    tumor_scores = probabilities[:, tumor_column]
    fpr, tpr, _ = roc_curve(y_train, tumor_scores, pos_label="tumor")
    model_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {model_auc:.3f})")

plt.plot([0, 1], [0, 1], "r--", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("Tumor Recall")
plt.title("Cross-Validated ROC Curves")
plt.legend()
plt.show()

# Ajuste-fino Random Forest 
params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

random_search = RandomizedSearchCV(
    rf_model,
    param_distributions=params,
    n_iter=5,
    cv=inner_cv,
    scoring="roc_auc",
    random_state=42,
    n_jobs=1,
)
random_search.fit(X_train, y_train, groups=groups_train)

best_rf = random_search.best_estimator_
print("Best Random Forest parameters:", random_search.best_params_)

# Evaluacion final del modelo en el conjunto de test
best_rf.fit(X_train, y_train)

test_predictions = best_rf.predict(X_test)
test_probabilities = best_rf.predict_proba(X_test)
tumor_column = list(best_rf.classes_).index("tumor")
tumor_scores = test_probabilities[:, tumor_column]

print("\nTuned Random Forest - test results")
print(f"ROC AUC: {roc_auc_score(y_test, tumor_scores):.4f}")
print("Confusion matrix (rows: actual, columns: predicted):")
print(confusion_matrix(y_test, test_predictions, labels=["normal", "tumor"]))
print(classification_report(y_test, test_predictions, labels=["normal", "tumor"]))

fpr, tpr, _ = roc_curve(y_test, tumor_scores, pos_label="tumor")
test_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"Tuned Random Forest (AUC = {test_auc:.3f})")
plt.plot([0, 1], [0, 1], "r--", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("Tumor Recall")
plt.title("Random Forest ROC Curve on Test Set")
plt.legend()
plt.show()

# Sanity checks

permuted_auc = []

for seed in range(10):
    rng = np.random.default_rng(seed)
    shuffled_y = rng.permutation(y_train)

    permuted_results = cross_validate(
        rf_model,
        X_train,
        shuffled_y,
        cv=inner_cv,
        groups=groups_train,
        scoring="roc_auc",
    )

    permuted_auc.append(permuted_results["test_score"].mean())

print("AUC with shuffled labels:", permuted_auc)
print("Mean:", np.mean(permuted_auc))

# Feature importance y DEGs mas significativos

importance = pd.Series(
    best_rf.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print(importance.head(20))

deg_data = pd.read_csv(r"C:\Users\alexm\Desktop\proyectos\rnaseq-brca-pipeline\rnaseq-brca-pipeline\data\processed\deg_brca.csv", index_col=0)
top_degs = deg_data.sort_values("padj").head(20)

top_importance_genes = importance.head(20).index
common_genes = set(top_importance_genes) & set(top_degs.index)
print("Common genes between top importance and top DEGs:", common_genes)

top_importance = importance.head(20).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(top_importance.index, top_importance.values, color="steelblue")
ax.set_xlabel("Random Forest feature importance")
ax.set_ylabel("Gene")
ax.set_title("Top 20 genes by Random Forest importance")
plt.tight_layout()
plt.savefig(FIGURES / "top20_rf_feature_importance.png", dpi=150)
plt.show()

deg_data.to_excel(PROCESSED / "deg_brca.xlsx", index=True)