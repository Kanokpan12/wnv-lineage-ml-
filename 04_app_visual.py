
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

from shiny import App, render, ui

# ----------------------------------------------------------------------------
# CONFIG -- adjust if your Script 1-3 outputs live elsewhere
# ----------------------------------------------------------------------------
OUT_DIR = "/Users/nakarinpamornchainavakul/Desktop/Data4publish/out/wnv_pipeline"

GENOTYPE_LEADERBOARD_PATH = f"{OUT_DIR}/genotype_model_comparison.csv"
GENOTYPE_POSITION_PATH = f"{OUT_DIR}/genotype_position_summary.csv"
GENOTYPE_GENE_PATH = f"{OUT_DIR}/genotype_gene_summary.csv"

PHENOTYPE_LEADERBOARD_PATH = f"{OUT_DIR}/phenotype_model_comparison.csv"
PHENOTYPE_AAGROUP_LEADERBOARD_PATH = f"{OUT_DIR}/phenotype_aagroup_model_comparison.csv"
PHENOTYPE_PERMUTATION_PATH = f"{OUT_DIR}/phenotype_permutation_test.json"
PHENOTYPE_AAGROUP_PERMUTATION_PATH = f"{OUT_DIR}/phenotype_aagroup_permutation_test.json"
PHENOTYPE_CV_METRICS_PATH = f"{OUT_DIR}/phenotype_repeated_cv_metrics.csv"
PHENOTYPE_AAGROUP_CV_METRICS_PATH = f"{OUT_DIR}/phenotype_aagroup_repeated_cv_metrics.csv"
PHENOTYPE_POSITION_PATH = f"{OUT_DIR}/phenotype_position_summary.csv"
PHENOTYPE_GENE_PATH = f"{OUT_DIR}/phenotype_gene_summary.csv"

MODEL_ABBREV = {
    "rf": "RF", "et": "ET", "dt": "DT", "ada": "AdaBoost",
    "gbc": "GBC", "lightgbm": "LGBM", "xgboost": "XGB",
}
METRIC_COLORS = {"Accuracy": "#377EB8", "AUC": "#E41A1C", "Recall": "#4DAF4A", "Prec.": "#FF7F00"}
METRIC_LABELS = {"Accuracy": "Accuracy", "AUC": "AUC", "Recall": "Recall", "Prec.": "Precision (PPV)"}

GENE_COLOR_CACHE = {}


def gene_color(gene):
    if gene not in GENE_COLOR_CACHE:
        cmap = plt.get_cmap("tab20c")
        idx = len(GENE_COLOR_CACHE) % 20
        GENE_COLOR_CACHE[gene] = matplotlib.colors.to_hex(cmap(idx / 20))
    return GENE_COLOR_CACHE[gene]


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_data():
    data = {
        "genotype_leaderboard": pd.read_csv(GENOTYPE_LEADERBOARD_PATH, index_col=0),
        "genotype_position": pd.read_csv(GENOTYPE_POSITION_PATH, index_col=0).sort_index(),
        "genotype_gene": pd.read_csv(GENOTYPE_GENE_PATH),
        "phenotype_leaderboard": pd.read_csv(PHENOTYPE_LEADERBOARD_PATH, index_col=0),
        "phenotype_aagroup_leaderboard": pd.read_csv(PHENOTYPE_AAGROUP_LEADERBOARD_PATH, index_col=0),
        "phenotype_cv_metrics": pd.read_csv(PHENOTYPE_CV_METRICS_PATH),
        "phenotype_aagroup_cv_metrics": pd.read_csv(PHENOTYPE_AAGROUP_CV_METRICS_PATH),
        "phenotype_position": pd.read_csv(PHENOTYPE_POSITION_PATH, index_col=0).sort_index(),
        "phenotype_gene": pd.read_csv(PHENOTYPE_GENE_PATH),
    }
    with open(PHENOTYPE_PERMUTATION_PATH) as f:
        data["phenotype_permutation"] = json.load(f)
    with open(PHENOTYPE_AAGROUP_PERMUTATION_PATH) as f:
        data["phenotype_aagroup_permutation"] = json.load(f)
    return data


DATA = load_data()


# ----------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------
def contiguous_gene_blocks(genes_in_order):
    start = 0
    n = len(genes_in_order)
    for i in range(1, n + 1):
        if i == n or genes_in_order[i] != genes_in_order[start]:
            yield start, i, genes_in_order[start]
            start = i


def draw_gene_banner(ax, genes_in_order, n_pos):
    ax.set_xlim(-0.5, n_pos - 0.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    for start, end, gene in contiguous_gene_blocks(genes_in_order):
        width = end - start
        ax.add_patch(Rectangle((start - 0.5, 0), width, 1, color=gene_color(gene), alpha=0.85))
        if width >= max(20, n_pos * 0.01):
            ax.text(start + width / 2 - 0.5, 0.5, gene, ha="center", va="center", fontsize=7)


def genes_in_genomic_order(genes_in_order):
    """Unique gene names in first-occurrence (i.e. genomic) order."""
    seen = []
    for g in genes_in_order:
        if g not in seen:
            seen.append(g)
    return seen


def leaderboard_plot(ax, leaderboard_df, title):
    """Horizontal grouped bars: y = model abbreviation, x = 0-1, 4 colored
    bars per model (Accuracy, AUC, Recall, Precision)."""
    df = leaderboard_df.copy()
    df["abbrev"] = [MODEL_ABBREV.get(i, i) for i in df.index]
    df = df.sort_values("AUC", ascending=True)  # best model at top

    metrics = ["Accuracy", "AUC", "Recall", "Prec."]
    n_models = len(df)
    y = np.arange(n_models)
    bar_h = 0.8 / len(metrics)

    for i, metric in enumerate(metrics):
        offset = (i - (len(metrics) - 1) / 2) * bar_h
        ax.barh(y + offset, df[metric], height=bar_h,
                color=METRIC_COLORS[metric], label=METRIC_LABELS[metric])

    ax.set_yticks(y)
    ax.set_yticklabels(df["abbrev"])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Performance")
    ax.set_title(title, fontsize=11, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=len(metrics),
              frameon=False, fontsize=8)


def dual_axis_gene_bars(ax, gene_df, gene_order, raw_col, mean_col, raw_color, mean_color, ylabel_metric):
    """Per-gene raw (sum) bar on the left axis and length-standardized (mean)
    bar on a twin right axis, side by side at each gene's x position."""
    gene_df = gene_df.set_index("gene").reindex(gene_order)
    x = np.arange(len(gene_order))
    width = 0.38

    ax.bar(x - width / 2, gene_df[raw_col], width=width, color=raw_color, label="Raw (sum)")
    ax.set_ylabel(f"{ylabel_metric}\n(raw sum)", color=raw_color)
    ax.tick_params(axis="y", labelcolor=raw_color)

    ax2 = ax.twinx()
    ax2.bar(x + width / 2, gene_df[mean_col], width=width, color=mean_color, alpha=0.8,
            label="Standardized (mean/position)")
    ax2.set_ylabel(f"{ylabel_metric}\n(mean per position)", color=mean_color)
    ax2.tick_params(axis="y", labelcolor=mean_color)

    ax.set_xticks(x)
    ax.set_xticklabels(gene_order, rotation=45, ha="right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.30),
              ncol=2, fontsize=8, frameon=False)


def gradient_bar(ax, x, values, cmap_name="RdBu_r", width=1.0):
    """Bars colored on a blue(low)->red(high) gradient by value."""
    vmax = max(values.max(), 1e-9)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(norm(values))
    ax.bar(x, values, width=width, color=colors)


# ----------------------------------------------------------------------------
# PAGE 1: Model performance
# ----------------------------------------------------------------------------
def plot_page1_performance():
    fig, axes = plt.subplots(3, 1, figsize=(10, 14))
    leaderboard_plot(axes[0], DATA["genotype_leaderboard"], "Genotype -- nucleotide-encoded models")
    leaderboard_plot(axes[1], DATA["phenotype_leaderboard"], "Phenotype -- nucleotide-encoded models")
    leaderboard_plot(axes[2], DATA["phenotype_aagroup_leaderboard"], "Phenotype -- amino-acid-group-encoded models")
    fig.suptitle("Model performance summary (compare_models leaderboards)", fontsize=13)
    fig.subplots_adjust(hspace=0.7, bottom=0.06, top=0.94)
    return fig


# Bar width for the full-genome position plots (pages 2 and 4). Raise this if
# the bars still look too thin -- at ~10,000 positions compressed into one
# figure width, a value of 1.0 renders as hairlines; try 3-6 for a bolder look.
# All positions in the genome are plotted (not just a top-N subset) -- most
# read as 0 since they were dropped by NZV filtering / never given to the
# model, which is itself part of the intended story (see page titles below).
POSITION_BAR_WIDTH = 4.0


# ----------------------------------------------------------------------------
# PAGE 2: Genotype position-wise
# ----------------------------------------------------------------------------
def plot_page2_genotype_position():
    pos_df = DATA["genotype_position"]
    genes_in_order = pos_df["gene"].tolist()
    n_pos = len(pos_df)
    x = np.arange(n_pos)

    fig = plt.figure(figsize=(16, 11))
    gs = gridspec.GridSpec(4, 1, height_ratios=[1.6, 1.6, 1.6, 0.5], hspace=0.35, figure=fig)

    ax_desc = fig.add_subplot(gs[0])
    ax_gain = fig.add_subplot(gs[1], sharex=ax_desc)
    ax_shap = fig.add_subplot(gs[2], sharex=ax_desc)
    ax_banner = fig.add_subplot(gs[3], sharex=ax_desc)

    gradient_bar(ax_desc, x, pos_df["freq_diff"].values, width=POSITION_BAR_WIDTH)
    ax_desc.set_ylabel("Freq. diff.")
    ax_desc.set_title("Descriptive between-lineage frequency difference by nt position "
                       "(color: blue=low -> red=high; all positions shown)", fontsize=10, loc="left")

    ax_gain.bar(x, pos_df["gain_max"], width=POSITION_BAR_WIDTH, color="#377EB8")
    ax_gain.set_ylabel("Gain\n(max per pos.)")
    ax_gain.set_title("Gain-based importance by nt position (all positions shown; "
                       "0 where the model never saw that position)", fontsize=10, loc="left")

    ax_shap.bar(x, pos_df["abs_shap_max"], width=POSITION_BAR_WIDTH, color="#E41A1C")
    ax_shap.set_ylabel("|SHAP|\n(max per pos.)")
    ax_shap.set_title("Mean |SHAP| by nt position, held-out test set (all positions shown; "
                       "0 where the model never saw that position)", fontsize=10, loc="left")

    draw_gene_banner(ax_banner, genes_in_order, n_pos)

    for a in [ax_desc, ax_gain, ax_shap]:
        a.set_xticks([])
        a.spines[["top", "right"]].set_visible(False)
        # free y-scale is the matplotlib default (autoscale per axes) -- no
        # ylim is set here, so each panel independently scales 0 to its own max

    fig.suptitle("Genotype: position-wise descriptive signal, gain, and SHAP (free y-scale per panel)",
                  fontsize=13, y=0.995)
    plt.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# PAGE 3: Genotype gene-level
# ----------------------------------------------------------------------------
def plot_page3_genotype_gene():
    pos_df = DATA["genotype_position"]
    gene_df = DATA["genotype_gene"]
    gene_order = genes_in_genomic_order(pos_df["gene"].tolist())
    gene_order = [g for g in gene_order if g in set(gene_df["gene"])]

    fig, axes = plt.subplots(3, 1, figsize=(12, 16))

    dual_axis_gene_bars(axes[0], gene_df, gene_order, "sum_freq_diff", "mean_freq_diff",
                        raw_color="#08306b", mean_color="#e31a1c", ylabel_metric="Freq. diff.")
    axes[0].set_title("Descriptive frequency-difference concentration by gene", fontsize=11, loc="left")

    dual_axis_gene_bars(axes[1], gene_df, gene_order, "sum_gain", "mean_gain",
                        raw_color="#08306b", mean_color="#6baed6", ylabel_metric="Gain")
    axes[1].set_title("Gain-based importance concentration by gene", fontsize=11, loc="left")

    dual_axis_gene_bars(axes[2], gene_df, gene_order, "sum_abs_shap", "mean_abs_shap",
                        raw_color="#a50f15", mean_color="#fc9272", ylabel_metric="|SHAP|")
    axes[2].set_title("Mean |SHAP| concentration by gene", fontsize=11, loc="left")

    fig.suptitle("Genotype: gene-level concentration -- raw (sum, left axis) vs. "
                 "length-standardized (mean/position, right axis)", fontsize=13, y=0.995)
    fig.subplots_adjust(hspace=0.9, bottom=0.05, top=0.94)
    return fig


# ----------------------------------------------------------------------------
# PAGE 4: Phenotype summary
# ----------------------------------------------------------------------------
def plot_page4_phenotype():
    pos_df = DATA["phenotype_position"]
    gene_df = DATA["phenotype_gene"]
    genes_in_order = pos_df["gene"].tolist()
    gene_order = genes_in_genomic_order(genes_in_order)
    gene_order = [g for g in gene_order if g in set(gene_df["gene"])]
    n_pos = len(pos_df)
    x = np.arange(n_pos)

    fig = plt.figure(figsize=(16, 20))
    gs = gridspec.GridSpec(
        6, 2, height_ratios=[1.4, 1.3, 1.4, 1.4, 1.8, 0.5], hspace=0.7, wspace=0.25, figure=fig
    )

    # -- permutation null (row 0) --
    ax_perm_nt = fig.add_subplot(gs[0, 0])
    ax_perm_aa = fig.add_subplot(gs[0, 1])
    for ax, perm, title in [
        (ax_perm_nt, DATA["phenotype_permutation"], "Nucleotide-encoded"),
        (ax_perm_aa, DATA["phenotype_aagroup_permutation"], "AA-group-encoded"),
    ]:
        ax.axvline(perm["observed_mean_auc"], color="#E41A1C", linewidth=2,
                   label=f"Observed (AUC={perm['observed_mean_auc']:.3f})")
        ax.axvspan(perm["null_mean_auc"] - perm["null_sd_auc"], perm["null_mean_auc"] + perm["null_sd_auc"],
                   color="#999999", alpha=0.3, label="Null mean +/- 1 SD")
        ax.axvline(perm["null_mean_auc"], color="#555555", linewidth=1, linestyle="--")
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Mean AUC")
        ax.set_title(f"{title}: observed vs. permutation null\n"
                     f"(p={perm['empirical_p_value']:.3f}, n_perm={perm['n_permutations']})",
                     fontsize=9, loc="left")
        ax.legend(fontsize=7, frameon=False, loc="upper right")
        ax.spines[["top", "right", "left"]].set_visible(False)

    # -- repeated-CV metric spread (row 1) --
    ax_cv_nt = fig.add_subplot(gs[1, 0])
    ax_cv_aa = fig.add_subplot(gs[1, 1])
    metric_cols = ["accuracy", "auc", "recall", "ppv", "f1"]
    for ax, cv_df, title in [
        (ax_cv_nt, DATA["phenotype_cv_metrics"], "Nucleotide-encoded"),
        (ax_cv_aa, DATA["phenotype_aagroup_cv_metrics"], "AA-group-encoded"),
    ]:
        ax.boxplot([cv_df[m].dropna() for m in metric_cols], labels=metric_cols, showmeans=True)
        ax.set_ylim(0, 1)
        ax.set_title(f"{title}: spread across {len(cv_df)} fold-evaluations", fontsize=9, loc="left")
        ax.spines[["top", "right"]].set_visible(False)

    # -- descriptive position-wise (row 2) --
    ax_desc = fig.add_subplot(gs[2, :])
    gradient_bar(ax_desc, x, pos_df["freq_diff"].values, width=POSITION_BAR_WIDTH)
    ax_desc.set_ylabel("Freq. diff.")
    ax_desc.set_title("Descriptive WNND-vs-non-WNND frequency difference by nt position "
                       "(color: blue=low -> red=high; all positions shown) -- no gain/SHAP "
                       "shown (see guide above)", fontsize=10, loc="left")
    ax_desc.set_xticks([])
    ax_desc.spines[["top", "right"]].set_visible(False)

    ax_banner = fig.add_subplot(gs[3, :], sharex=ax_desc)
    draw_gene_banner(ax_banner, genes_in_order, n_pos)

    # -- gene-level descriptive concentration (row 4) --
    ax_gene = fig.add_subplot(gs[4, :])
    dual_axis_gene_bars(ax_gene, gene_df, gene_order, "sum_freq_diff", "mean_freq_diff",
                        raw_color="#08306b", mean_color="#e31a1c", ylabel_metric="Freq. diff.")
    ax_gene.set_title("Gene-level descriptive concentration -- raw (sum) vs. length-standardized (mean)",
                       fontsize=10, loc="left")

    fig.suptitle("Phenotype (WNND): no detectable signal in either feature encoding", fontsize=13, y=0.998)
    return fig


PHENOTYPE_GUIDE_MD = """
**How to read this page**

- **Top row (permutation null):** the red line is the model's actual average performance (AUC);
  the gray band is where performance lands if WNND labels are randomly shuffled (i.e. pure chance),
  averaged over 100 shuffles. If the red line sits inside the gray band, the model cannot be
  distinguished from a model with no real information -- which is the case for both feature encodings
  here (p ~ 0.44-0.47, far from a conventional significance threshold).
- **Second row (CV spread):** each box summarizes the SAME metric recomputed across 100 different
  train/test splits (5 folds x 20 repeats). The wide boxes reflect how unstable any single estimate
  is at n=101 -- a single lucky or unlucky split could easily show a much higher or lower number than
  the true (near-chance) average.
- **Descriptive panels (bottom):** these come directly from the raw sequence data, with no model
  involved. Low, fairly flat values across the genome mean there isn't a strong individual position
  that separates WNND from non-WNND cases -- consistent with the permutation-test result above.
- No gain/SHAP importance panel is shown for phenotype: ranking "important" features from a model
  that did not beat chance would risk implying a finding that isn't actually there.
"""


# ----------------------------------------------------------------------------
# Shiny UI
# ----------------------------------------------------------------------------
app_ui = ui.page_fluid(
    ui.h3("WNV Final Results Dashboard"),
    ui.input_radio_buttons(
        "page", "",
        choices=["1. Model Performance", "2. Genotype: Position-wise",
                 "3. Genotype: Gene-level", "4. Phenotype Summary"],
        selected="1. Model Performance", inline=True,
    ),
    ui.output_ui("guide_text"),
    ui.output_plot("main_plot", height="1500px"),
)


def server(input, output, session):
    @output
    @render.ui
    def guide_text():
        if input.page() == "4. Phenotype Summary":
            return ui.markdown(PHENOTYPE_GUIDE_MD)
        return None

    @output
    @render.plot
    def main_plot():
        page = input.page()
        if page == "1. Model Performance":
            return plot_page1_performance()
        elif page == "2. Genotype: Position-wise":
            return plot_page2_genotype_position()
        elif page == "3. Genotype: Gene-level":
            return plot_page3_genotype_gene()
        else:
            return plot_page4_phenotype()


app = App(app_ui, server)