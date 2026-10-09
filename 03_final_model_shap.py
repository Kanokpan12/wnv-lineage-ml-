
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    RandomizedSearchCV, GridSearchCV, StratifiedGroupKFold, StratifiedKFold,
)
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, recall_score, precision_score, f1_score,
)
from sklearn.inspection import PartialDependenceDisplay

import shap

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
OUT_DIR = "/Users/nakarinpamornchainavakul/Desktop/Data4publish/out/wnv_pipeline"

GENOTYPE_ENCODED_PATH = f"{OUT_DIR}/genotype_encoded.parquet"
GENOTYPE_SIGNATURES_PATH = f"{OUT_DIR}/genotype_descriptive_signatures.csv"
PHENOTYPE_ENCODED_PATH = f"{OUT_DIR}/phenotype_encoded.parquet"
PHENOTYPE_AAGROUP_ENCODED_PATH = f"{OUT_DIR}/phenotype_aagroup_encoded.parquet"
BEST_MODELS_PATH = f"{OUT_DIR}/best_models.json"

RANDOM_STATE = 42

# --- Genotype ---
GENOTYPE_TIME_TEST_FRAC = 0.20   # most recent 20% of DATED samples -> test
GENOTYPE_CV_FOLDS = 5
GENOTYPE_SEARCH_ITER = 30
GENOTYPE_TOP_N_SHAP_FEATURES = 20

# --- Phenotype (both scopes) ---
PHENOTYPE_N_FOLDS = 5
PHENOTYPE_N_REPEATS = 20          # repeated GroupKFold for the final metric estimate
PHENOTYPE_N_PERMUTATIONS = 100    # label-shuffled null distribution (single 5-fold each)
PHENOTYPE_TUNING_SCORING = "average_precision"  # PR-AUC; more appropriate than ROC-AUC
                                                 # for modest imbalance + small n

K_CHOICES = {
    "phenotype": [20, 50, 100, 200, 300],
    "phenotype_aagroup": [20, 50, 100, 171],
}


# ----------------------------------------------------------------------------
# Model factory + regularization-heavy hyperparameter grids
# (small grids by design -- GridSearchCV is meant to be exhaustive-but-cheap
# here, not a broad search; p>>n means we WANT to bias toward simple models)
# ----------------------------------------------------------------------------
def make_model(model_id):
    factory = {
        "rf": lambda: RandomForestClassifier(random_state=RANDOM_STATE),
        "et": lambda: ExtraTreesClassifier(random_state=RANDOM_STATE),
        "dt": lambda: DecisionTreeClassifier(random_state=RANDOM_STATE),
        "ada": lambda: AdaBoostClassifier(random_state=RANDOM_STATE),
        "gbc": lambda: GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    if LGBMClassifier is not None:
        factory["lightgbm"] = lambda: LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)
    if XGBClassifier is not None:
        factory["xgboost"] = lambda: XGBClassifier(
            random_state=RANDOM_STATE, eval_metric="logloss", use_label_encoder=False
        )
    if model_id not in factory:
        raise ValueError(f"Unknown or unavailable model id: {model_id}")
    return factory[model_id]()


def get_clf_param_grid(model_id, y):
    """Small, heavily-regularized grids, prefixed 'clf__' for use inside a Pipeline."""
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    imbalance_ratio = n_neg / max(n_pos, 1)

    grids = {
        "xgboost": {
            "clf__max_depth": [2, 3],
            "clf__min_child_weight": [5, 10],
            "clf__subsample": [0.7, 1.0],
            "clf__scale_pos_weight": [1.0, imbalance_ratio],
            "clf__n_estimators": [150],
        },
        "lightgbm": {
            "clf__max_depth": [2, 3],
            "clf__min_child_samples": [5, 10],
            "clf__subsample": [0.7, 1.0],
            "clf__n_estimators": [150],
        },
        "gbc": {
            "clf__max_depth": [1, 2],
            "clf__min_samples_leaf": [5, 10],
            "clf__subsample": [0.7, 1.0],
        },
        "rf": {
            "clf__max_depth": [2, 3],
            "clf__min_samples_leaf": [3, 5],
            "clf__class_weight": [None, "balanced"],
            "clf__n_estimators": [300],
        },
        "et": {
            "clf__max_depth": [2, 3],
            "clf__min_samples_leaf": [3, 5],
            "clf__class_weight": [None, "balanced"],
            "clf__n_estimators": [300],
        },
        "dt": {
            "clf__max_depth": [2, 3],
            "clf__min_samples_leaf": [5, 10],
            "clf__class_weight": [None, "balanced"],
        },
        "ada": {
            "clf__n_estimators": [50, 150],
            "clf__learning_rate": [0.5, 1.0],
        },
    }
    if model_id not in grids:
        raise ValueError(f"No param grid defined for model id: {model_id}")
    return grids[model_id]


# ----------------------------------------------------------------------------
# Fisher's-exact-test univariate feature selection (fold-internal transformer)
# ----------------------------------------------------------------------------
def fisher_pvalues(X, y):
    """X: 2D binary array, y: 1D binary array. Returns per-column Fisher exact p-values."""
    y = np.asarray(y)
    idx1 = y == 1
    idx0 = y == 0
    a = X[idx1].sum(axis=0)              # feature=1, y=1
    b = int(idx1.sum()) - a              # feature=0, y=1
    c = X[idx0].sum(axis=0)              # feature=1, y=0
    d = int(idx0.sum()) - c              # feature=0, y=0

    pvals = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        _, p = fisher_exact([[a[j], b[j]], [c[j], d[j]]])
        pvals[j] = p
    return pvals


class FisherKBest(BaseEstimator, TransformerMixin):
    """Selects the k features with the smallest Fisher's-exact-test p-value.
    Fit fresh on every CV fold's training data only -- never on the full dataset."""

    def __init__(self, k=50):
        self.k = k

    def fit(self, X, y):
        X = np.asarray(X)
        pvals = fisher_pvalues(X, y)
        k = min(self.k, X.shape[1])
        self.selected_idx_ = np.argsort(pvals)[:k]
        self.pvals_ = pvals
        return self

    def transform(self, X):
        X = np.asarray(X)
        return X[:, self.selected_idx_]


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else np.nan,
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "ppv": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


# ----------------------------------------------------------------------------
# SHAP helper -- tries the fast, exact TreeExplainer first; falls back to a
# model-agnostic explainer (fine at these small feature counts) for models
# TreeExplainer doesn't support (e.g. AdaBoost in some shap/sklearn versions).
# ----------------------------------------------------------------------------
def compute_shap_values(clf, X):
    try:
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        elif hasattr(sv, "ndim") and sv.ndim == 3:
            sv = sv[:, :, 1]
        return np.asarray(sv)
    except Exception:
        background = X if X.shape[0] <= 50 else shap.sample(X, 50, random_state=0)
        explainer = shap.Explainer(clf.predict_proba, background)
        sv = explainer(X)
        vals = sv.values
        if vals.ndim == 3:
            vals = vals[:, :, 1]
        return np.asarray(vals)


# ----------------------------------------------------------------------------
# Genotype gene-density summary (descriptive, non-ML)
# ----------------------------------------------------------------------------
def plot_gene_signature_density(sig_csv_path):
    """
    For each level (nt, aa), computes what fraction of ALL positions
    annotated to a given gene were flagged as lineage-'signature' positions
    in Script 1 -- i.e. density, not raw count, so large and small genes are
    compared fairly. This is the honest replacement for a SHAP/importance
    ranking here: with ~2,000 largely redundant signature positions genome-
    wide, WHICH handful a tree model ranks highest is not biologically
    meaningful, but WHERE signature positions concentrate is.
    """
    df = pd.read_csv(sig_csv_path)
    df = df[df["gene"] != "NA"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, level, label in zip(axes, ["nt", "aa"], ["Nucleotide", "Amino acid"]):
        sub = df[df["level"] == level]
        grouped = sub.groupby("gene").agg(
            total=("position", "count"), signature=("is_signature", "sum")
        )
        grouped["density"] = grouped["signature"] / grouped["total"]
        grouped = grouped.sort_values("density", ascending=True)

        ax.barh(grouped.index, grouped["density"], color="#377EB8")
        ax.set_xlabel("Fraction of positions flagged as lineage-signature")
        ax.set_title(f"{label} level (n genes = {len(grouped)})")
        ax.set_xlim(0, 1.0)

        for i, (gene, row) in enumerate(grouped.iterrows()):
            ax.text(row["density"] + 0.01, i, f"{int(row['signature'])}/{int(row['total'])}",
                    va="center", fontsize=8)

    fig.suptitle("Genotype: where lineage-signature positions concentrate, by gene",
                  fontsize=13)
    plt.tight_layout()
    plt.show()

    return df


# ----------------------------------------------------------------------------
# Full-genome position/gene-level exports for the Script 4 Shiny dashboard
# ----------------------------------------------------------------------------
def load_full_position_table(level, sig_csv_path):
    """Full-genome per-position table (every position, not just NZV survivors),
    with gene assignment and descriptive freq-diff -- the shared x-axis backbone
    for the Script 4 dashboard."""
    df = pd.read_csv(sig_csv_path)
    sub = df[df["level"] == level][["position", "gene", "between_class_freq_diff"]].copy()
    sub = sub.rename(columns={"between_class_freq_diff": "freq_diff"})
    return sub.sort_values("position").set_index("position")


def position_level_max(feature_cols, values, index):
    """Reduce one-hot columns (e.g. 'nt_121_A') to one max value per position,
    0 for positions absent from feature_cols (i.e. dropped by NZV filtering)."""
    pos_best = {}
    for c, v in zip(feature_cols, values):
        pos = int(c.split("_")[1])
        if pos not in pos_best or v > pos_best[pos]:
            pos_best[pos] = v
    return pd.Series([pos_best.get(p, 0.0) for p in index], index=index)


def export_genotype_position_and_gene_summaries(feature_cols, gain_values, shap_values_matrix):
    full_table = load_full_position_table("nt", GENOTYPE_SIGNATURES_PATH)

    full_table["gain_max"] = position_level_max(feature_cols, gain_values, full_table.index)
    mean_abs_shap_per_col = np.abs(shap_values_matrix).mean(axis=0)
    full_table["abs_shap_max"] = position_level_max(feature_cols, mean_abs_shap_per_col, full_table.index)

    pos_path = f"{OUT_DIR}/genotype_position_summary.csv"
    full_table.to_csv(pos_path)
    print(f"-> wrote {pos_path}")

    gene_table = full_table[full_table["gene"] != "NA"].groupby("gene").agg(
        n_positions=("gene", "size"),
        sum_freq_diff=("freq_diff", "sum"),
        mean_freq_diff=("freq_diff", "mean"),
        sum_gain=("gain_max", "sum"),
        mean_gain=("gain_max", "mean"),
        sum_abs_shap=("abs_shap_max", "sum"),
        mean_abs_shap=("abs_shap_max", "mean"),
    ).reset_index()

    gene_path = f"{OUT_DIR}/genotype_gene_summary.csv"
    gene_table.to_csv(gene_path, index=False)
    print(f"-> wrote {gene_path}")


def export_phenotype_descriptive_summaries():
    """Descriptive-only (no ML) position/gene summaries for the phenotype
    dashboard page -- there is no gain/SHAP to include, by design (see
    module docstring)."""
    full_table = load_full_position_table("nt", f"{OUT_DIR}/phenotype_descriptive_signatures.csv")

    pos_path = f"{OUT_DIR}/phenotype_position_summary.csv"
    full_table.to_csv(pos_path)
    print(f"-> wrote {pos_path}")

    gene_table = full_table[full_table["gene"] != "NA"].groupby("gene").agg(
        n_positions=("gene", "size"),
        sum_freq_diff=("freq_diff", "sum"),
        mean_freq_diff=("freq_diff", "mean"),
    ).reset_index()

    gene_path = f"{OUT_DIR}/phenotype_gene_summary.csv"
    gene_table.to_csv(gene_path, index=False)
    print(f"-> wrote {gene_path}")


# ==============================================================================
# GENOTYPE
# ==============================================================================
def run_genotype(model_id):
    print("\n" + "=" * 70)
    print("GENOTYPE -- formal train/test, tuning, SHAP")
    print("=" * 70)

    df = pd.read_parquet(GENOTYPE_ENCODED_PATH)
    feature_cols = [c for c in df.columns if c not in ("Accession", "target", "year")]

    dated = df[df["year"].notna()].sort_values("year")
    undated = df[df["year"].isna()]
    n_test = int(len(dated) * GENOTYPE_TIME_TEST_FRAC)
    test_df = dated.iloc[-n_test:]
    train_df = pd.concat([dated.iloc[:-n_test], undated], axis=0)

    print(f"Train: n={len(train_df)} (includes {len(undated)} undated) | "
          f"Test: n={len(test_df)} (most recent dated records, "
          f"years {test_df['year'].min():.0f}-{test_df['year'].max():.0f})")
    print(f"Train class counts: {train_df['target'].value_counts().to_dict()}")
    print(f"Test class counts:  {test_df['target'].value_counts().to_dict()}")

    X_train, y_train = train_df[feature_cols].values, train_df["target"].values
    X_test, y_test = test_df[feature_cols].values, test_df["target"].values

    param_dist = {
        "n_estimators": [200, 400, 600],
        "max_depth": [None, 6, 10, 16],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", 0.3, 0.5],
        "class_weight": [None, "balanced"],
    }
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_distributions=param_dist,
        n_iter=GENOTYPE_SEARCH_ITER,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=GENOTYPE_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    print(f"\nBest RF params: {search.best_params_}")
    print(f"Best CV ROC-AUC (train): {search.best_score_:.4f}")

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)
    print("\nHeld-out test performance:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    pd.DataFrame([metrics]).to_csv(f"{OUT_DIR}/genotype_test_metrics.csv", index=False)

    # ---- Gene-density summary (the primary interpretation for genotype --
    #      see module docstring for why a SHAP ranking is not meaningful here) ----
    print("\nGene-level signature density (descriptive, from Script 1's "
          f"{len(pd.read_csv(GENOTYPE_SIGNATURES_PATH))}-position signature table)...")
    plot_gene_signature_density(GENOTYPE_SIGNATURES_PATH)

    # ---- Gain-based importance and SHAP: kept only as supporting evidence
    #      FOR the redundancy argument above (note how flat the top features'
    #      magnitudes are) -- NOT presented as a ranked list of "the important
    #      sites". See module docstring. ----
    importances = pd.Series(best_model.feature_importances_, index=feature_cols)
    top_importance = importances.sort_values(ascending=False).head(GENOTYPE_TOP_N_SHAP_FEATURES)

    fig, ax = plt.subplots(figsize=(8, 6))
    top_importance.iloc[::-1].plot.barh(ax=ax, color="#377EB8")
    ax.set_xlabel("Gain-based importance")
    ax.set_title(f"Genotype: top {GENOTYPE_TOP_N_SHAP_FEATURES} features (Random Forest)")
    plt.tight_layout()
    plt.show()

    # ---- SHAP on the held-out test set ----
    print("\nComputing SHAP on the held-out test set...")
    shap_values = compute_shap_values(best_model, X_test)
    explanation = shap.Explanation(
        values=shap_values, data=X_test, feature_names=feature_cols
    )

    shap.plots.beeswarm(explanation, max_display=GENOTYPE_TOP_N_SHAP_FEATURES, show=False)
    plt.title("Genotype: SHAP beeswarm (held-out test set)")
    plt.tight_layout()
    plt.show()

    # ---- Export full-genome position/gene summaries for the Script 4 dashboard ----
    export_genotype_position_and_gene_summaries(feature_cols, importances.values, shap_values)

    top_shap_features = (
        pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols)
        .sort_values(ascending=False)
        .head(4)
        .index.tolist()
    )
    fig, axes = plt.subplots(1, len(top_shap_features), figsize=(4 * len(top_shap_features), 4))
    PartialDependenceDisplay.from_estimator(
        best_model, X_train, features=[feature_cols.index(f) for f in top_shap_features],
        feature_names=feature_cols, ax=axes,
    )
    plt.suptitle("Genotype: partial dependence, top SHAP features")
    plt.tight_layout()
    plt.show()

    return {"model": best_model, "metrics": metrics, "feature_cols": feature_cols}


# ==============================================================================
# PHENOTYPE (nucleotide and aa-group -- same function, different inputs)
# ==============================================================================
def run_phenotype_scope(name, encoded_path, model_id, k_choices):
    print("\n" + "=" * 70)
    print(f"{name.upper()} -- fold-internal Fisher selection, tuning, repeated CV")
    print("(no SHAP/importance -- see module docstring for rationale)")
    print("=" * 70)

    df = pd.read_parquet(encoded_path)
    feature_cols = [c for c in df.columns if c not in ("Accession", "target", "Organization")]
    X = df[feature_cols].values
    y = df["target"].values
    groups = df["Organization"].values

    print(f"n = {len(df)} | class counts: {pd.Series(y).value_counts().to_dict()} | "
          f"{df['Organization'].nunique()} groups | {len(feature_cols)} candidate features")

    # ---- Step 1: tune ONCE (pragmatic, non-nested) ----
    pipe = Pipeline([("select", FisherKBest()), ("clf", make_model(model_id))])
    param_grid = {"select__k": k_choices, **get_clf_param_grid(model_id, y)}

    tuning_cv = StratifiedGroupKFold(n_splits=PHENOTYPE_N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        pipe, param_grid, scoring=PHENOTYPE_TUNING_SCORING,
        cv=tuning_cv, n_jobs=-1, refit=False,
    )
    grid.fit(X, y, groups=groups)
    best_params = grid.best_params_
    print(f"\nBest params (tuned once via GroupKFold, scoring={PHENOTYPE_TUNING_SCORING}): {best_params}")
    print(f"Best tuning CV score: {grid.best_score_:.4f}")

    # ---- Step 2: repeated GroupKFold evaluation with FIXED hyperparameters ----
    all_metrics = []

    for repeat in range(PHENOTYPE_N_REPEATS):
        splitter = StratifiedGroupKFold(n_splits=PHENOTYPE_N_FOLDS, shuffle=True, random_state=repeat)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
            fold_pipe = clone(pipe).set_params(**best_params)
            fold_pipe.fit(X[train_idx], y[train_idx])

            y_pred = fold_pipe.predict(X[test_idx])
            y_proba = fold_pipe.predict_proba(X[test_idx])[:, 1]
            m = compute_metrics(y[test_idx], y_pred, y_proba)
            m.update({"repeat": repeat, "fold": fold, "n_test": len(test_idx)})
            all_metrics.append(m)

    metrics_df = pd.DataFrame(all_metrics)
    summary = metrics_df[["accuracy", "auc", "recall", "ppv", "f1"]].agg(["mean", "std"])
    print(f"\nRepeated GroupKFold evaluation ({PHENOTYPE_N_REPEATS} repeats x "
          f"{PHENOTYPE_N_FOLDS} folds = {len(metrics_df)} fold-evaluations):")
    print(summary.to_string())

    metrics_df.to_csv(f"{OUT_DIR}/{name}_repeated_cv_metrics.csv", index=False)
    summary.to_csv(f"{OUT_DIR}/{name}_repeated_cv_summary.csv")

    # ---- Permutation (label-shuffling) null distribution ----
    print(f"\nRunning {PHENOTYPE_N_PERMUTATIONS} label-shuffled permutations "
          f"(single {PHENOTYPE_N_FOLDS}-fold GroupKFold each, fixed hyperparameters)...")
    observed_auc = summary.loc["mean", "auc"]
    null_aucs = []
    rng = np.random.default_rng(RANDOM_STATE)
    for p in range(PHENOTYPE_N_PERMUTATIONS):
        y_shuffled = rng.permutation(y)
        splitter = StratifiedGroupKFold(n_splits=PHENOTYPE_N_FOLDS, shuffle=True, random_state=p)
        fold_aucs = []
        for train_idx, test_idx in splitter.split(X, y_shuffled, groups):
            if len(np.unique(y_shuffled[train_idx])) < 2 or len(np.unique(y_shuffled[test_idx])) < 2:
                continue
            perm_pipe = clone(pipe).set_params(**best_params)
            perm_pipe.fit(X[train_idx], y_shuffled[train_idx])
            proba = perm_pipe.predict_proba(X[test_idx])[:, 1]
            fold_aucs.append(roc_auc_score(y_shuffled[test_idx], proba))
        if fold_aucs:
            null_aucs.append(np.mean(fold_aucs))
    null_aucs = np.array(null_aucs)

    empirical_p = (1 + np.sum(null_aucs >= observed_auc)) / (1 + len(null_aucs))
    print(f"Observed mean AUC: {observed_auc:.4f}")
    print(f"Null distribution (label-shuffled): mean={null_aucs.mean():.4f}, "
          f"sd={null_aucs.std():.4f}")
    print(f"Empirical p-value (observed vs. null): {empirical_p:.4f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(null_aucs, bins=20, color="#999999", alpha=0.8, label="Label-shuffled null")
    ax.axvline(observed_auc, color="#E41A1C", linewidth=2, label=f"Observed (AUC={observed_auc:.3f})")
    ax.set_xlabel("Mean AUC")
    ax.set_ylabel("Count")
    ax.set_title(f"{name}: observed performance vs. permutation null (p={empirical_p:.3f})")
    ax.legend()
    plt.tight_layout()
    plt.show()

    with open(f"{OUT_DIR}/{name}_permutation_test.json", "w") as f:
        json.dump({
            "observed_mean_auc": float(observed_auc),
            "null_mean_auc": float(null_aucs.mean()),
            "null_sd_auc": float(null_aucs.std()),
            "empirical_p_value": float(empirical_p),
            "n_permutations": len(null_aucs),
        }, f, indent=2)

    return {"metrics_summary": summary, "best_params": best_params, "permutation_p": empirical_p}


# ==============================================================================
def main():
    with open(BEST_MODELS_PATH) as f:
        best_models = json.load(f)

    genotype_results = run_genotype(best_models["genotype"]["model_id"])

    export_phenotype_descriptive_summaries()

    phenotype_results = run_phenotype_scope(
        "phenotype", PHENOTYPE_ENCODED_PATH,
        best_models["phenotype"]["model_id"], K_CHOICES["phenotype"],
    )

    phenotype_aagroup_results = run_phenotype_scope(
        "phenotype_aagroup", PHENOTYPE_AAGROUP_ENCODED_PATH,
        best_models["phenotype_aagroup"]["model_id"], K_CHOICES["phenotype_aagroup"],
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Genotype test AUC: {genotype_results['metrics']['auc']:.4f}")
    print(f"Phenotype (nt) mean AUC: {phenotype_results['metrics_summary'].loc['mean','auc']:.4f}, "
          f"permutation p={phenotype_results['permutation_p']:.4f}")
    print(f"Phenotype (aa-group) mean AUC: "
          f"{phenotype_aagroup_results['metrics_summary'].loc['mean','auc']:.4f}, "
          f"permutation p={phenotype_aagroup_results['permutation_p']:.4f}")

    print("\nScript 3 complete.")


if __name__ == "__main__":
    main()