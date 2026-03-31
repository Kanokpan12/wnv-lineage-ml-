import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.cluster import KMeans

# -------------------------------
# File paths
# -------------------------------
dist_file = "file path /03_cleaned_sequences.fasta.mldist"

# -------------------------------
# Read IQ-TREE mldist
# -------------------------------
with open(dist_file, "r") as f:
    lines = f.readlines()

n = int(lines[0].strip())
sample_names = []
dist_values = []

for line in lines[1:]:
    parts = line.strip().split()
    sample_names.append(parts[0])
    dist_values.append([float(x) for x in parts[1:]])

dist_matrix = np.array(dist_values)

# -------------------------------
# Run PCoA (classical via MDS)
# -------------------------------
mds = MDS(
    n_components=2,
    dissimilarity="precomputed",
    random_state=42,
    n_jobs=-1
)

coords = mds.fit_transform(dist_matrix)

# -------------------------------
# Variance explained (approx)
# -------------------------------
H = np.eye(n) - np.ones((n, n)) / n
B = -0.5 * H @ (dist_matrix ** 2) @ H
eigvals = np.sort(np.linalg.eigvalsh(B))[::-1]
var_explained = eigvals[:2] / np.sum(eigvals[eigvals > 0]) * 100

# -------------------------------
# Create PCoA dataframe
# -------------------------------
pcoa_df = pd.DataFrame({
    "Sample": sample_names,
    "PCoA1": coords[:, 0],
    "PCoA2": coords[:, 1]
})

# -------------------------------
# K-means clustering (k = 3)
# -------------------------------
kmeans = KMeans(n_clusters=3, random_state=42, n_init=25)
pcoa_df["Cluster"] = kmeans.fit_predict(pcoa_df[["PCoA1", "PCoA2"]])

# -------------------------------
# Assign biological lineage labels
# (manual but reproducible)
# -------------------------------
cluster_centers = pd.DataFrame(
    kmeans.cluster_centers_,
    columns=["PCoA1", "PCoA2"]
)

# Australia = most separated cluster (largest norm)
australia_cluster = np.argmax(
    np.sqrt(cluster_centers["PCoA1"]**2 + cluster_centers["PCoA2"]**2)
)

lineage_map = {}
lin_num = 1
for c in range(3):
    if c == australia_cluster:
        lineage_map[c] = "Lineage_3_Australia"
    else:
        lineage_map[c] = f"Lineage_{lin_num}"
        lin_num += 1

pcoa_df["Lineage"] = pcoa_df["Cluster"].map(lineage_map)

# -------------------------------
# Remove specific sequences by name
# -------------------------------
sequences_to_remove = ['FJ159129', 'FJ159130', 'FJ159131', 'AY277251', 'AY765264']
pcoa_clean = pcoa_df[~pcoa_df["Sample"].isin(sequences_to_remove)].copy()

print(f"\nRemoved {len(sequences_to_remove)} specific sequences")
print("Removed sequences:", sequences_to_remove)

# -------------------------------
# Filter out sequences with PCoA2 < -0.6
# -------------------------------
n_before = len(pcoa_clean)
pcoa_clean = pcoa_clean[pcoa_clean["PCoA2"] >= -0.6].copy()
n_removed = n_before - len(pcoa_clean)

print(f"\nRemoved {n_removed} sequences with PCoA2 < -0.6")

# -------------------------------
# Within-lineage genetic distance
# -------------------------------
within_results = []

for lin, sub in pcoa_clean.groupby("Lineage"):
    idx = sub.index.to_numpy()
    d = dist_matrix[np.ix_(idx, idx)]
    vals = d[np.triu_indices_from(d, k=1)]

    within_results.append({
        "Lineage": lin,
        "N": len(sub),
        "Mean_within_distance": np.mean(vals)
    })

within_df = pd.DataFrame(within_results)

# -------------------------------
# Between-lineage genetic distance
# -------------------------------
between_results = []

lineages = pcoa_clean["Lineage"].unique()

for i in range(len(lineages)):
    for j in range(i + 1, len(lineages)):
        l1, l2 = lineages[i], lineages[j]

        idx1 = pcoa_clean[pcoa_clean["Lineage"] == l1].index
        idx2 = pcoa_clean[pcoa_clean["Lineage"] == l2].index

        d = dist_matrix[np.ix_(idx1, idx2)]

        between_results.append({
            "Lineage_1": l1,
            "Lineage_2": l2,
            "Mean_between_distance": np.mean(d)
        })

between_df = pd.DataFrame(between_results)

# -------------------------------
# Plot PCoA
# -------------------------------
plt.figure(figsize=(10, 8))

for lin in pcoa_clean["Lineage"].unique():
    s = pcoa_clean[pcoa_clean["Lineage"] == lin]
    plt.scatter(
        s["PCoA1"], s["PCoA2"],
        s=80, alpha=0.75, label=f"{lin} (n={len(s)})"
    )

plt.xlabel(f"PCoA1 ({var_explained[0]:.2f}%)", fontsize=12)
plt.ylabel(f"PCoA2 ({var_explained[1]:.2f}%)", fontsize=12)
plt.title("WNV PCoA after lineage assignment", fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# -------------------------------
# Output results
# -------------------------------
print("\nWithin-lineage distances:")
print(within_df)

print("\nBetween-lineage distances:")
print(between_df)