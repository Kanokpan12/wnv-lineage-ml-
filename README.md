# wnv-lineage-ml

**Machine Learning–Based Prediction of West Nile Virus Lineages and Associated Pathogenic Risk** (unpublished)  

This repository provides a **complete computational and experimental workflow** for studying West Nile virus (WNV) genomic and lineage prediction. The analyses integrate phylogenetics, machine learning, and evolutionary analysis.  

## Workflow Overview

The project is organized into four main stages/folders:  

1. **STAGE 1: Phylogenetic Labeling & Data Preparation**  
   * Sequence cleaning, tree construction, and lineage mapping  
   * Outputs: cleaned sequences, phylogenetic trees, annotated lineage labels  

2. **STAGE 2: Machine Learning–Based Lineage Classification**  
   * Whole-genome and gene-level predictive modeling using PyCaret  
   * Feature importance mapping and gene-level visualization  
   * Outputs: trained ML models, performance metrics, feature-gene importance maps  

3. **STAGE 3: Selective Pressure Analysis**  
   * Entropy screening, dN/dS estimation, and host-stratified evolutionary analysis  
   * Outputs: site-specific selective pressure estimates, gene-level plots  

4. **Miscellaneous: Additional R Analyses**  
   * Epidemiology: US state-level WNV case mapping (historical 2009–2018 and recent 2025)  
   * Mouse survival study: weight and clinical score visualization  
   * Outputs: epidemiological maps, mouse study figures  

## Folder Structure

Each folder contains:  

* Scripts for analysis (`.R` and `.py`)  
* Input data files (`.csv`)  
* Instructions to reproduce the analyses  

## Key Highlights

* End-to-end workflow integrating **Python + R**  
* Phylogenetic labeling and cluster-supported lineage assignment  
* Supervised machine learning for lineage prediction  
* Evolutionary analysis including selective pressure estimation  
* Visualization of genomic, epidemiological, and in vivo experimental data  

---

**Note:** All analyses are reproducible. Please ensure required Python and R packages are installed before running the scripts.  

