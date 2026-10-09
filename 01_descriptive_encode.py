import numpy as np
import pandas as pd
from Bio import SeqIO

# ----------------------------------------------------------------------------
# CONFIG -- adjust paths / thresholds here
# ----------------------------------------------------------------------------
FASTA_PATH = "/Users/nakarinpamornchainavakul/Desktop/Data4publish/ML_WNVSeq.fasta"
PROTEINREF_PATH = "/Users/nakarinpamornchainavakul/Desktop/Data4publish/proteinRef.tsv"
GENOTYPE_META_PATH = "/Users/nakarinpamornchainavakul/Desktop/Data4publish/WNV_metadata_genotype.csv"
PHENOTYPE_META_PATH = "/Users/nakarinpamornchainavakul/Desktop/Data4publish/WNV_metadata_phenotype.csv"

OUT_DIR = "/Users/nakarinpamornchainavakul/Desktop/Data4publish/out/wnv_pipeline"

# Descriptive "signature position" thresholds
MIN_WITHIN_CLASS_CONSERVATION = 0.90   # majority-allele frequency required in EACH class
MIN_BETWEEN_CLASS_FREQ_DIFF = 0.50     # min |freq difference| for the divergent allele

# NZV filtering: a binary (0/1) one-hot column is dropped only if it is
# near-constant across the WHOLE dataset (ignoring class) AND does not differ
# meaningfully in prevalence between classes. The whole-dataset check alone
# already retains genuine class discriminators (their overall frequency
# reflects the class split, so they aren't constant dataset-wide); the
# between-class rescue additionally protects rare variants confined to a
# small subset within just one class, which the whole-dataset check can miss
# purely due to overall rarity.
NZV_FREQ_RATIO_CUTOFF = 19.0  # ~95:5, whole-dataset variability check
NZV_MIN_CLASS_DIFF_RESCUE = 0.05  # min |freq1 - freq0| that rescues an otherwise-NZV column

NT_BASES = ["A", "T", "C", "G"]  # + "N" bucket (covers N, gaps, IUPAC ambiguity codes)

# 7-way Zappo-style biochemical grouping for amino acids (matches the palette
# grouping used in app_signature_viewer.py), plus an "Other" bucket absorbing
# stop (*), ambiguous/indel-disrupted (X), and gap (-) -- mirrors how the nt
# scheme buckets everything non-ACGT into a single "N" category.
AA_GROUP_MAP = {
    "I": "Aliphatic", "L": "Aliphatic", "V": "Aliphatic", "A": "Aliphatic", "M": "Aliphatic",
    "F": "Aromatic", "W": "Aromatic", "Y": "Aromatic",
    "K": "Positive", "R": "Positive", "H": "Positive",
    "D": "Negative", "E": "Negative",
    "S": "Polar", "T": "Polar", "N": "Polar", "Q": "Polar",
    "P": "Special", "G": "Special",
    "C": "Cysteine",
}
AA_GROUP_ORDER = ["Aliphatic", "Aromatic", "Positive", "Negative", "Polar", "Special", "Cysteine", "Other"]

STANDARD_CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


# ----------------------------------------------------------------------------
# Loading helpers
# ----------------------------------------------------------------------------
def load_alignment(fasta_path):
    """Returns (accessions: list[str], seq_array: np.ndarray[n_seq, n_pos] dtype='S1')."""
    records = list(SeqIO.parse(fasta_path, "fasta"))
    accessions = [r.id for r in records]
    n_pos = len(records[0].seq)
    arr = np.empty((len(records), n_pos), dtype="S1")
    for i, r in enumerate(records):
        s = str(r.seq).upper().encode("ascii")
        arr[i, :] = np.frombuffer(s, dtype="S1")
    return accessions, arr


def load_protein_ref(path):
    """Returns (ref_df, lookup) where lookup[aa_position] -> protein name (1-based, 0 unused)."""
    ref = pd.read_csv(path, sep="\t")
    max_aa = int(ref["stop"].max())
    lookup = np.full(max_aa + 1, "NA", dtype=object)
    for _, row in ref.iterrows():
        lookup[int(row["start"]): int(row["stop"]) + 1] = row["protein"]
    return ref, lookup


def translate_alignment(seq_array):
    """
    Codon-aligned translation of the whole MSA (column-based, NOT per-sequence
    degapped translation) so amino-acid columns stay comparable across
    sequences. A codon with 1-2 gap characters (an indel-disrupted codon in
    this alignment) is translated as 'X' (ambiguous); an all-gap codon is
    translated as '-' (clean in-frame deletion); any codon containing a
    non-ACGT/non-gap IUPAC ambiguity code is also translated as 'X'.
    """
    n_seq, n_pos = seq_array.shape
    n_codons = n_pos // 3
    aa_array = np.empty((n_seq, n_codons), dtype="U1")
    codons_view = seq_array[:, : n_codons * 3].reshape(n_seq, n_codons, 3)

    for i in range(n_seq):
        row = codons_view[i]
        for c in range(n_codons):
            codon = (row[c, 0] + row[c, 1] + row[c, 2]).decode("ascii")
            gap_count = codon.count("-")
            if gap_count == 3:
                aa_array[i, c] = "-"
            elif gap_count > 0:
                aa_array[i, c] = "X"
            else:
                aa_array[i, c] = STANDARD_CODON_TABLE.get(codon, "X")
    return aa_array


# ----------------------------------------------------------------------------
# Descriptive signature analysis
# ----------------------------------------------------------------------------
def signature_table(matrix, positions, target, class_labels, level, gene_lookup=None):
    """
    matrix: np.ndarray[n_samples, n_positions] of single-character symbols (bytes or str)
    positions: 1-based position numbers matching matrix columns
    target: array-like of 0/1 (aligned to matrix rows)
    class_labels: dict {0: name0, 1: name1}
    level: "nt" or "aa"
    gene_lookup: array mapping an aa-position -> gene/protein name
    """
    target = np.asarray(target)
    idx0 = np.where(target == 0)[0]
    idx1 = np.where(target == 1)[0]
    n0, n1 = len(idx0), len(idx1)

    rows = []
    for j, pos in enumerate(positions):
        col = matrix[:, j]
        if col.dtype.kind == "S":
            col = np.array([c.decode("ascii") for c in col])

        col0 = col[idx0]
        col1 = col[idx1]

        vals0, counts0 = np.unique(col0, return_counts=True)
        vals1, counts1 = np.unique(col1, return_counts=True)

        maj0_i = np.argmax(counts0)
        maj1_i = np.argmax(counts1)
        maj0_allele, maj0_freq = vals0[maj0_i], counts0[maj0_i] / n0
        maj1_allele, maj1_freq = vals1[maj1_i], counts1[maj1_i] / n1

        freq0_map = dict(zip(vals0, counts0 / n0))
        freq1_map = dict(zip(vals1, counts1 / n1))
        all_alleles = set(freq0_map) | set(freq1_map)
        freq_diff = max(
            abs(freq0_map.get(a, 0.0) - freq1_map.get(a, 0.0)) for a in all_alleles
        )

        is_signature = (
            maj0_freq >= MIN_WITHIN_CLASS_CONSERVATION
            and maj1_freq >= MIN_WITHIN_CLASS_CONSERVATION
            and maj0_allele != maj1_allele
            and freq_diff >= MIN_BETWEEN_CLASS_FREQ_DIFF
        )

        gene = "NA"
        if level == "nt" and gene_lookup is not None:
            aa_pos = (pos - 1) // 3 + 1
            gene = gene_lookup[aa_pos] if aa_pos < len(gene_lookup) else "NA"
        elif level == "aa" and gene_lookup is not None:
            gene = gene_lookup[pos] if pos < len(gene_lookup) else "NA"

        rows.append({
            "level": level,
            "position": int(pos),
            "gene": gene,
            f"majority_{class_labels[0]}": maj0_allele,
            f"freq_{class_labels[0]}": round(maj0_freq, 4),
            f"majority_{class_labels[1]}": maj1_allele,
            f"freq_{class_labels[1]}": round(maj1_freq, 4),
            "between_class_freq_diff": round(freq_diff, 4),
            "is_signature": is_signature,
        })

    df = pd.DataFrame(rows)
    return df.sort_values(
        ["is_signature", "between_class_freq_diff"], ascending=[False, False]
    ).reset_index(drop=True)


def composition_table(matrix, positions, target, class_labels, level, gene_lookup):
    """
    Full allele/residue frequency composition (not just the majority) per
    position per class -- long format, used for the stacked-bar visualization.
    Columns: level, position, gene, class, allele, freq
    """
    target = np.asarray(target)
    idx_by_class = {0: np.where(target == 0)[0], 1: np.where(target == 1)[0]}

    rows = []
    for pos in positions:
        j = int(pos) - 1
        col = matrix[:, j]
        if col.dtype.kind == "S":
            col = np.array([c.decode("ascii") for c in col])

        if level == "nt":
            aa_pos = (pos - 1) // 3 + 1
            gene = gene_lookup[aa_pos] if aa_pos < len(gene_lookup) else "NA"
        else:
            gene = gene_lookup[pos] if pos < len(gene_lookup) else "NA"

        for cls_idx in (0, 1):
            sub = col[idx_by_class[cls_idx]]
            n = len(sub)
            vals, counts = np.unique(sub, return_counts=True)
            for v, c in zip(vals, counts):
                rows.append({
                    "level": level,
                    "position": int(pos),
                    "gene": gene,
                    "class": class_labels[cls_idx],
                    "allele": v,
                    "freq": c / n,
                })
    return pd.DataFrame(rows)


def select_positions(sig_df, mode):
    """mode: 'signature_only' (genotype) or 'top100' (phenotype)."""
    df = sig_df.sort_values(
        ["is_signature", "between_class_freq_diff"], ascending=[False, False]
    )
    if mode == "signature_only":
        return df[df["is_signature"]]["position"].tolist()
    elif mode == "top100":
        return df.head(100)["position"].tolist()
    raise ValueError(f"Unknown selection mode: {mode}")


# ----------------------------------------------------------------------------
# One-hot encoding + stratified NZV filtering
# ----------------------------------------------------------------------------
def one_hot_encode(seq_array):
    """
    seq_array: np.ndarray[n_seq, n_pos] dtype='S1'
    Returns (feature_matrix: np.ndarray[n_seq, n_pos*5] int8, colnames: list[str])
    Column order per position: A, T, C, G, N(other/gap)
    """
    n_seq, n_pos = seq_array.shape
    letters = NT_BASES  # A T C G
    blocks = []
    colnames = []

    known = np.zeros((n_seq, n_pos), dtype=bool)
    for letter in letters:
        b = seq_array == letter.encode("ascii")
        blocks.append(b)
        known |= b
    other = ~known  # N, gap, and any IUPAC ambiguity code

    for pos in range(1, n_pos + 1):
        for letter in letters:
            colnames.append(f"nt_{pos}_{letter}")
        colnames.append(f"nt_{pos}_N")

    stacked = np.stack(blocks + [other], axis=-1)  # (n_seq, n_pos, 5)
    feature_matrix = stacked.reshape(n_seq, n_pos * 5).astype(np.int8)
    return feature_matrix, colnames


def stratified_nzv_filter(feature_matrix, target, freq_ratio_cutoff=NZV_FREQ_RATIO_CUTOFF,
                           min_class_diff_rescue=NZV_MIN_CLASS_DIFF_RESCUE):
    """
    Keep a binary column if EITHER:
      (a) it has adequate variability across the WHOLE dataset (majority:minority
          ratio <= freq_ratio_cutoff, ignoring class) -- the standard NZV check.
          Note: if a position is conserved to a DIFFERENT fixed value in each
          class (e.g. 100% A in class 1, 100% T in class 0), the whole-dataset
          column is naturally NOT constant (its overall frequency reflects the
          class split), so this check alone already keeps genuine class
          discriminators -- no per-class computation needed for that case.
      (b) its prevalence (freq of value=1) differs between class 0 and class 1
          by at least min_class_diff_rescue -- a rescue for variants that are
          real but confined to a small subset WITHIN one class (e.g. present
          in only 3/37 WNND cases and 0/64 non-WNND), which can fail the
          whole-dataset check (a) purely because of their overall rarity.

    A column is dropped only if it is near-constant dataset-wide AND shows no
    meaningful difference in prevalence between classes -- i.e. it is
    genuinely uninformative, not just globally rare.

    Returns a boolean mask of columns to keep.
    """
    n = feature_matrix.shape[0]
    ones = feature_matrix.sum(axis=0)
    zeros = n - ones
    majority = np.maximum(ones, zeros)
    minority = np.minimum(ones, zeros)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(minority == 0, np.inf, majority / np.maximum(minority, 1))
    globally_variable = ratio <= freq_ratio_cutoff

    target = np.asarray(target)
    freq0 = feature_matrix[target == 0].mean(axis=0)
    freq1 = feature_matrix[target == 1].mean(axis=0)
    class_discriminating = np.abs(freq1 - freq0) >= min_class_diff_rescue

    return globally_variable | class_discriminating


def one_hot_encode_aa_groups(aa_array):
    """
    aa_array: np.ndarray[n_seq, n_codons] dtype='U1' (single-letter aa symbols,
    plus '*' stop, 'X' ambiguous/indel-disrupted, '-' gap)
    Returns (feature_matrix: np.ndarray[n_seq, n_codons*8] int8, colnames: list[str])
    Column order per residue position: Aliphatic, Aromatic, Positive, Negative,
    Polar, Special, Cysteine, Other (stop/X/gap).
    """
    n_seq, n_codons = aa_array.shape
    group_masks = {g: np.zeros((n_seq, n_codons), dtype=bool) for g in AA_GROUP_ORDER}

    for letter in np.unique(aa_array):
        group = AA_GROUP_MAP.get(letter, "Other")
        group_masks[group] |= (aa_array == letter)

    colnames = []
    for pos in range(1, n_codons + 1):
        for g in AA_GROUP_ORDER:
            colnames.append(f"aa_{pos}_{g}")

    stacked = np.stack([group_masks[g] for g in AA_GROUP_ORDER], axis=-1)  # (n_seq, n_codons, 8)
    feature_matrix = stacked.reshape(n_seq, n_codons * len(AA_GROUP_ORDER)).astype(np.int8)
    return feature_matrix, colnames


# ----------------------------------------------------------------------------
# Category pipeline
# ----------------------------------------------------------------------------
def run_category(name, meta_path, target_col, accessions, seq_array, aa_array,
                  gene_lookup, extra_group_cols, composition_selection_mode,
                  also_build_aa_group_encoding=False):
    print(f"\n=== {name.upper()} ===")
    meta = pd.read_csv(meta_path)
    meta = meta[meta["Accession"].isin(accessions)].copy()

    acc_to_idx = {a: i for i, a in enumerate(accessions)}
    meta["_row"] = meta["Accession"].map(acc_to_idx)
    meta = meta.dropna(subset=["_row", target_col]).copy()
    meta["_row"] = meta["_row"].astype(int)

    y_raw = meta[target_col]
    if pd.api.types.is_numeric_dtype(y_raw):
        y = y_raw.astype(int).values
        classes_sorted = sorted(pd.Series(y).unique())
        class_labels = {0: str(classes_sorted[0]), 1: str(classes_sorted[1])}
    else:
        classes_sorted = sorted(y_raw.astype(str).unique())
        mapping = {classes_sorted[0]: 0, classes_sorted[1]: 1}
        y = y_raw.astype(str).map(mapping).values
        class_labels = {0: str(classes_sorted[0]), 1: str(classes_sorted[1])}

    rows_idx = meta["_row"].values
    sub_nt = seq_array[rows_idx]
    sub_aa = aa_array[rows_idx]

    print(f"n = {len(rows_idx)} | class counts: {pd.Series(y).value_counts().to_dict()}")

    # ---- Descriptive signature tables ----
    nt_positions = np.arange(1, sub_nt.shape[1] + 1)
    aa_positions = np.arange(1, sub_aa.shape[1] + 1)

    nt_sig = signature_table(sub_nt, nt_positions, y, class_labels, "nt", gene_lookup)
    aa_sig = signature_table(sub_aa, aa_positions, y, class_labels, "aa", gene_lookup)
    combined_sig = pd.concat([nt_sig, aa_sig], ignore_index=True)

    sig_path = f"{OUT_DIR}/{name}_descriptive_signatures.csv"
    combined_sig.to_csv(sig_path, index=False)
    print(f"Signature positions flagged: {int(combined_sig['is_signature'].sum())} "
          f"(nt: {int(nt_sig['is_signature'].sum())}, aa: {int(aa_sig['is_signature'].sum())})")
    print(f"-> wrote {sig_path}")

    # ---- Full composition table (for stacked-bar visualization) ----
    nt_positions_selected = select_positions(nt_sig, composition_selection_mode)
    aa_positions_selected = select_positions(aa_sig, composition_selection_mode)

    comp_nt = composition_table(sub_nt, nt_positions_selected, y, class_labels, "nt", gene_lookup)
    comp_aa = composition_table(sub_aa, aa_positions_selected, y, class_labels, "aa", gene_lookup)
    composition = pd.concat([comp_nt, comp_aa], ignore_index=True)

    comp_path = f"{OUT_DIR}/{name}_composition.csv"
    composition.to_csv(comp_path, index=False)
    print(f"Composition data: {len(nt_positions_selected)} nt positions, "
          f"{len(aa_positions_selected)} aa positions -> wrote {comp_path}")

    # ---- One-hot encode + stratified NZV filter ----
    feat_matrix, colnames = one_hot_encode(sub_nt)
    print(f"One-hot features before NZV filtering: {feat_matrix.shape[1]}")

    keep_mask = stratified_nzv_filter(feat_matrix, y)
    feat_matrix = feat_matrix[:, keep_mask]
    colnames = [c for c, k in zip(colnames, keep_mask) if k]
    print(f"One-hot features after stratified NZV filtering: {feat_matrix.shape[1]}")

    encoded = pd.DataFrame(feat_matrix, columns=colnames)
    encoded.insert(0, "Accession", meta["Accession"].values)
    encoded["target"] = y
    for col in extra_group_cols:
        encoded[col] = meta[col].values if col in meta.columns else np.nan

    out_path = f"{OUT_DIR}/{name}_encoded.parquet"
    encoded.to_parquet(out_path, index=False)
    print(f"-> wrote {out_path}  (shape={encoded.shape})")

    # ---- OPTIONAL standalone comparison feature set: amino-acid biochemical
    #      groups instead of raw nucleotides (7-way Zappo scheme + "Other" for
    #      stop/ambiguous/gap). Built from sub_aa, independent of the nt
    #      encoding above -- meant to be compared head-to-head against it, not
    #      combined with it. ----
    if also_build_aa_group_encoding:
        aa_feat_matrix, aa_colnames = one_hot_encode_aa_groups(sub_aa)
        print(f"AA-group one-hot features before NZV filtering: {aa_feat_matrix.shape[1]}")

        aa_keep_mask = stratified_nzv_filter(aa_feat_matrix, y)
        aa_feat_matrix = aa_feat_matrix[:, aa_keep_mask]
        aa_colnames = [c for c, k in zip(aa_colnames, aa_keep_mask) if k]
        print(f"AA-group one-hot features after stratified NZV filtering: {aa_feat_matrix.shape[1]}")

        aa_encoded = pd.DataFrame(aa_feat_matrix, columns=aa_colnames)
        aa_encoded.insert(0, "Accession", meta["Accession"].values)
        aa_encoded["target"] = y
        for col in extra_group_cols:
            aa_encoded[col] = meta[col].values if col in meta.columns else np.nan

        aa_out_path = f"{OUT_DIR}/{name}_aagroup_encoded.parquet"
        aa_encoded.to_parquet(aa_out_path, index=False)
        print(f"-> wrote {aa_out_path}  (shape={aa_encoded.shape})")

    return combined_sig, encoded


def get_year(date_str):
    if pd.isna(date_str):
        return None
    s = str(date_str)
    parts = s.split("/")
    if len(parts) == 3:
        yy = parts[2]
        return 2000 + int(yy) if len(yy) == 2 else int(yy)
    if len(s) == 4 and s.isdigit():
        return int(s)
    return None


def main():
    print("Loading alignment...")
    accessions, seq_array = load_alignment(FASTA_PATH)
    print(f"Loaded {len(accessions)} sequences x {seq_array.shape[1]} positions")

    print("Loading protein reference...")
    _, gene_lookup = load_protein_ref(PROTEINREF_PATH)

    print("Translating alignment (codon-aligned)...")
    aa_array = translate_alignment(seq_array)
    print(f"Translated to {aa_array.shape[1]} codons/residues")

    # ---- Genotype ----
    geno_meta = pd.read_csv(GENOTYPE_META_PATH)
    geno_meta["year"] = geno_meta["Collection_Date"].apply(get_year)
    geno_meta_path = f"{OUT_DIR}/_genotype_meta_with_year.csv"
    geno_meta.to_csv(geno_meta_path, index=False)

    run_category(
        "genotype", geno_meta_path, "Lineage",
        accessions, seq_array, aa_array, gene_lookup,
        extra_group_cols=["year"],
        composition_selection_mode="signature_only",
    )

    # ---- Phenotype ----
    pheno_meta = pd.read_csv(PHENOTYPE_META_PATH)
    pheno_meta_path = f"{OUT_DIR}/_phenotype_meta.csv"
    pheno_meta.to_csv(pheno_meta_path, index=False)

    run_category(
        "phenotype", pheno_meta_path, "WNND",
        accessions, seq_array, aa_array, gene_lookup,
        extra_group_cols=["Organization"],
        composition_selection_mode="top100",
        also_build_aa_group_encoding=True,
    )

    print("\nScript 1 complete.")


if __name__ == "__main__":
    main()