from Bio import SeqIO
import pandas as pd
from itertools import combinations

# -----------------------------
# File paths
# -----------------------------
folder_path = "file path "
input_fasta = f"{folder_path}/02_WNV_UTR_removed.fasta"
output_fasta = f"{folder_path}/03_cleaned_sequences.fasta"
fail_report_file = f"{folder_path}/failed_sequences_report.csv"
metadata_file = "file path/WNV_completed.csv"

# -----------------------------
# Parameters
# -----------------------------
max_N_percent = 10
identity_threshold = 1

# -----------------------------
# Read metadata
# -----------------------------
metadata_df = pd.read_csv(metadata_file)

# -----------------------------
# Read fasta
# -----------------------------
fasta_records = list(SeqIO.parse(input_fasta, "fasta"))
if len(fasta_records) == 0:
    raise ValueError("No sequences found in the input FASTA.")

# -----------------------------
# Calculate N% per sequence
# -----------------------------
seq_data = []
for rec in fasta_records:
    seq = str(rec.seq).upper()
    length = len(seq)
    N_count = seq.count("N")
    N_percent = (N_count / length) * 100
    seq_data.append({
        "Accession": rec.id,
        "SeqLength": length,
        "N_percent": N_percent,
        "FailReason": None,
        "SeqRecord": rec
    })

df = pd.DataFrame(seq_data)

# -----------------------------
# Filter sequences with >10% N
# -----------------------------
df.loc[df["N_percent"] > max_N_percent, "FailReason"] = ">10% N"

# Keep sequences passing N% filter
keep_df = df[df["FailReason"].isna()].copy()

# -----------------------------
# Remove identical sequences (100% identity)
# -----------------------------
to_remove_indices = set()
for i, j in combinations(range(len(keep_df)), 2):
    seq1 = str(keep_df.iloc[i]["SeqRecord"].seq).upper()
    seq2 = str(keep_df.iloc[j]["SeqRecord"].seq).upper()
    min_len = min(len(seq1), len(seq2))
    seq1_trim = seq1[:min_len]
    seq2_trim = seq2[:min_len]
    matches = sum(a==b for a,b in zip(seq1_trim, seq2_trim))
    identity = matches / min_len
    if identity >= identity_threshold:
        to_remove_indices.add(keep_df.index[j])

keep_df.loc[keep_df.index.isin(to_remove_indices), "FailReason"] = "Identical"

# -----------------------------
# Final cleaned sequences
# -----------------------------
cleaned_df = keep_df[keep_df["FailReason"].isna()].copy()
SeqIO.write(cleaned_df["SeqRecord"], output_fasta, "fasta")

# -----------------------------
# Merge fail report with metadata
# -----------------------------
fail_df = df[df["FailReason"].notna()].copy()
fail_df = fail_df.drop(columns="SeqRecord")
fail_df_merged = fail_df.merge(metadata_df, on="Accession", how="left")

# Save fail report
fail_df_merged.to_csv(fail_report_file, index=False)

# -----------------------------
print(f"Cleaning finished!")
print(f"{len(cleaned_df)} sequences retained.")
print(f"{len(fail_df_merged)} sequences failed QC and saved in report.")
print(f"Cleaned FASTA saved at: {output_fasta}")
print(f"Fail report saved at: {fail_report_file}")
