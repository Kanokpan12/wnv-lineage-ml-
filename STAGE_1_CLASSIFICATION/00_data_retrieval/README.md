# Classification Workflow

This directory contains scripts used for retrieving, preprocessing, and annotating sequence data for the West Nile virus (WNV) lineage classification analysis. 
The workflow prepares phylogenetically informed labels for downstream machine learning models.

## Overview

The data retrieval and labeling pipeline consists of the following steps:

### Step 1: Sequence Preprocessing

**Script:** `04_Cleaning_ambigous_identity.py`

This script removes sequences with ambiguous nucleotide identities and performs basic quality control filtering to ensure that only high-confidence sequences are retained for phylogenetic analysis.

**Output:**

* Cleaned sequence dataset suitable for phylogenetic tree construction

---

### Step 2: Phylogenetic Tree Construction

**Tool:** IQ-TREE

Cleaned WNV sequences are used to construct a maximum-likelihood phylogenetic tree using IQ-TREE.

**Example command:**

```
iqtree2 -s input_alignment.fasta -m TEST -bb 1000 -alrt 1000
```

**Output:**

* Maximum-likelihood phylogenetic tree file
* Bootstrap support values for branch confidence

---

### Step 3: Mapping Accession Numbers to Tree Branches

**Script:** `05_Phylo_Label.R`

This script maps accession numbers to their corresponding phylogenetic branches and assigns labels based on tree topology.

**Output:**

* Annotated phylogenetic tree with accession-level labels

---

### Step 4: Lineage Identification Using Clustering and Ordination

**Script:** `06_PCoAPlot.py`

Principal Coordinates Analysis (PCoA) and K-means clustering are applied to genetic distance matrices to identify lineage structure across phylogenetic branches. These analyses help confirm lineage assignments inferred from tree topology.

**Output:**

* PCoA visualization of lineage structure
* Cluster-based lineage assignments

---

### Step 5: Genetic Distance Calculation Between Groups

**Script:** `07_Genetic_distance.py`

This script calculates pairwise genetic distances between lineage groups to quantify sequence similarity and support lineage-level separation.

**Output:**

* Pairwise genetic distance matrix
* Summary statistics describing inter-lineage similarity

---

## Expected Outputs of This Module

After completing this workflow, the following resources will be available for downstream classification tasks:

* Cleaned and filtered sequence dataset
* Maximum-likelihood phylogenetic tree
* Accession-level annotations
* Cluster-supported lineage labels
* Genetic distance summaries between lineage groups

These outputs serve as the foundation for generating reliable training labels used in machine learning–based lineage classification.

---

## Notes

* Sequence alignment was performed using MAFFT prior to phylogenetic tree construction with IQ-TREE (default settings).
* Scripts are designed to run sequentially in the order listed above.
* Intermediate files from earlier steps are required for downstream analyses.

For reproducibility, record software versions used for IQ-TREE, Python, and R packages when executing this pipeline.

