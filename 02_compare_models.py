import json
import pandas as pd
from pycaret.classification import ClassificationExperiment

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
OUT_DIR = "/Users/nakarinpamornchainavakul/Desktop/Data4publish/out/wnv_pipeline"

GENOTYPE_ENCODED_PATH = f"{OUT_DIR}/genotype_encoded.parquet"
PHENOTYPE_ENCODED_PATH = f"{OUT_DIR}/phenotype_encoded.parquet"
PHENOTYPE_AAGROUP_ENCODED_PATH = f"{OUT_DIR}/phenotype_aagroup_encoded.parquet"

SESSION_ID = 42
N_FOLDS = 5

GENOTYPE_SORT_METRIC = "Accuracy"
PHENOTYPE_SORT_METRIC = "Accuracy"  # sensitivity; switch to "Prec." for PPV

# Tree-based models only (matches the compare_models scope agreed for this
# project -- fast on wide sparse one-hot data, native gain-based importance,
# and compatible with SHAP's exact TreeExplainer in Script 3).
#   dt        Decision Tree
#   rf        Random Forest
#   et        Extra Trees
#   ada       AdaBoost (tree-stump based)
#   gbc       Gradient Boosting Classifier
#   lightgbm  LightGBM
#   xgboost   XGBoost (requires `pip install xgboost`; PyCaret will raise
#             "Estimator xgboost Not Available" if it's not installed --
#             either install it, or leave it out of this list as below)
TREE_MODELS = ["dt", "rf", "et", "ada", "gbc", "lightgbm", "xgboost"]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def run_comparison(name, data_path, sort_metric, fold_strategy="stratifiedkfold",
                    fold_groups=None, extra_ignore_features=None):
    print(f"\n=== {name.upper()} : compare_models ===")
    df = pd.read_parquet(data_path)

    ignore_features = ["Accession"] + (extra_ignore_features or [])
    ignore_features = [c for c in ignore_features if c in df.columns]

    exp = ClassificationExperiment()
    setup_kwargs = dict(
        data=df,
        target="target",
        session_id=SESSION_ID,
        ignore_features=ignore_features,
        fold=N_FOLDS,
        fold_strategy=fold_strategy,
        # Data is already fully engineered (filtered one-hot, no missing
        # values) -- disable preprocessing steps that are unnecessary and,
        # given the wide feature matrix, potentially very expensive (e.g. an
        # O(p^2) correlation matrix for multicollinearity removal).
        normalize=False,
        transformation=False,
        remove_multicollinearity=False,
        feature_selection=False,
        remove_outliers=False,
        verbose=False,
        html=False,
    )
    if fold_groups is not None:
        setup_kwargs["fold_groups"] = fold_groups

    exp.setup(**setup_kwargs)

    exp.compare_models(include=TREE_MODELS, sort=sort_metric, n_select=1)
    results = exp.pull()

    results_path = f"{OUT_DIR}/{name}_model_comparison.csv"
    results.to_csv(results_path)
    print(results.to_string())
    print(f"-> wrote {results_path}")

    best_id = results.index[0]
    best_display_name = results.iloc[0]["Model"]
    best_score = float(results.iloc[0][sort_metric])

    print(f"Best model for {name}: id='{best_id}' ({best_display_name}), "
          f"{sort_metric}={best_score:.4f}")

    return {
        "model_id": best_id,
        "display_name": best_display_name,
        "sort_metric": sort_metric,
        "score": best_score,
        "ignore_features": ignore_features,
    }


def main():
    best = {}

    best["genotype"] = run_comparison(
        "genotype", GENOTYPE_ENCODED_PATH,
        sort_metric=GENOTYPE_SORT_METRIC,
        fold_strategy="stratifiedkfold",
        extra_ignore_features=["year"],
    )

    best["phenotype"] = run_comparison(
        "phenotype", PHENOTYPE_ENCODED_PATH,
        sort_metric=PHENOTYPE_SORT_METRIC,
        fold_strategy="groupkfold",
        fold_groups="Organization",
        extra_ignore_features=["Organization"],
    )

    # Standalone comparison: amino-acid biochemical-group features (7-way
    # Zappo scheme) instead of raw nucleotide one-hot, for phenotype only.
    # Kept as a SEPARATE run (not combined with the nt feature set) so the
    # two encodings can be compared head-to-head.
    best["phenotype_aagroup"] = run_comparison(
        "phenotype_aagroup", PHENOTYPE_AAGROUP_ENCODED_PATH,
        sort_metric=PHENOTYPE_SORT_METRIC,
        fold_strategy="groupkfold",
        fold_groups="Organization",
        extra_ignore_features=["Organization"],
    )

    best_path = f"{OUT_DIR}/best_models.json"
    with open(best_path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"\n-> wrote {best_path}")
    print("\nScript 2 complete.")


if __name__ == "__main__":
    main()