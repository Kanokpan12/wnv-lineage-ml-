# Machine Learning Workflow

   <img width="804" height="481" alt="image" src="https://github.com/user-attachments/assets/fdbf0b57-7556-45a4-a6da-006390e7698c" />


This module contains machine learning workflows used to classify West Nile virus (WNV) lineage from genomic sequence features.

# Part 1: Machine Learning on Whole Genome Sequences (WGS)
## Step 1: Lineage Label Assignment

Sequences were assigned lineage labels based on K-means clustering results using the reference file:

08_Lineage_assignments.tsv

## Step 2: Model Training and Evaluation

Script: 18_ML_Dtermine.py

Supervised machine learning models were trained using PyCaret to predict lineage from whole-genome sequence–derived features.

**Output:**

* Model comparison results
* Cross-validation performance metrics
* Selected best-performing models
## Step 3: Feature Importance and Gene Mapping

Script: 20_ML_features.py

This step evaluates model performance across classifiers and maps important nucleotide positions to gene annotations.

**Output:**

* Feature importance summary
* Gene-level mapping results stored in: 21_Importance_mapped.csv

## Step 4: Visualization of Important Genomic Regions

Visualization highlights nucleotide positions contributing most strongly to lineage prediction and their correspondence to annotated genes.

# Part 2: Machine Learning on Individual Genes
## Step 1: Gene-Level Dataset Preparation

Gene regions were extracted using reference coordinates from: 15_start_stop_proteinRef

## Step 2: Gene-Level Model Evaluation

Machine learning models were trained separately on individual gene segments to evaluate predictive contribution and importance gain from each gene.

**Output:**

* Gene-specific model performance comparisons
* Relative importance of genes for lineage classification
