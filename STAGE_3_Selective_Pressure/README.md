# Selective Pressure Analysis Workflow

This module evaluates evolutionary selective pressure across West Nile virus (WNV) genes using entropy-based variability screening and downstream dN/dS analysis.

## Part 1: Gene-Level Selective Pressure Analysis

### Step 1: Identify Highly Variable Genes

**Script:** `16_Entropy_WNV.R`

Shannon entropy was calculated across genomic positions to identify genes with high sequence variability.

**Output:**

* Ranked variability scores across genes

### Step 2: Select Candidate Genes

The top three genes with the highest variability were selected for downstream selective pressure analysis.

### Step 3: Sequence Cleaning and Gene Splitting

**Scripts:**

* `19_cleanseq_dn_ds.R`
* `23_Split_genes.R`

Sequences were cleaned and separated into gene-specific datasets suitable for codon-based analysis.

### Step 4: Final CDS Quality Control

**Script:** `24_CDS_clean.R`

Coding sequences were finalized after additional filtering to ensure correct reading frames and removal of problematic sequences.

### Step 5: Selective Pressure Analysis

Selective pressure (dN/dS) analysis was performed using the Datamonkey server.

**Output:**

* Site-level selective pressure estimates
* Evidence of positive, neutral, and negative selection

### Step 6: Visualization of dN/dS Results

**Scripts:**

* `25_dnds_plot.py`
* `41_dnds_monomer.py`

These scripts generate plots summarizing selective pressure signals across genes.

---

## Part 2: Selective Pressure Analysis Across Hosts

Selective pressure patterns were further evaluated across different host groups.

### Step 1: Host-Based Sequence Mapping

Sequence accession numbers were mapped to three selected genes and host categories using:

`4._host_acccession.csv`

**Output:**

* Host-stratified gene datasets
* Comparative selective pressure summaries across host groups
