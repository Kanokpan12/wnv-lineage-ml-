# Additional Analyses: Epidemiology and Mouse Study Visualization

This folder contains R-based scripts for epidemiological analysis of West Nile virus (WNV) and visualization of mouse survival study data. The workflows support exploratory data analysis, visualization, and comparative interpretation of temporal and spatial trends.

# Epidemiology Analysis
## Step 1: Historical WNV Cases (2009–2018)
Data: Epi.csv — average WNV case counts by US state (2009–2018)
Script: EpiMap.R — plots state-level frequency maps

**Output:**
* Choropleth map showing historical WNV case distribution across the US

## Step 2: Recent WNV Cases (2025)
Data: Epi.csv — WNV case reports for 2025 by US state
Script: Epi2025.R — plots 2025 case distribution

**Output:**
* State-level case frequency visualization for 2025

# Mouse Survival Study Visualization
## Step 1: Mouse Weight Analysis
Data: Weight.csv — longitudinal weight measurements of mice
Script: WeightPlot.R — visualizes weight changes over time
## Step 2: Clinical Score Analysis
Data: ClinicalScore.csv — clinical severity scores from survival study
Script: ClinicalScore.R — visualizes clinical progression trends

**Output:**
Time-series plots for mouse weight and clinical scores to assess WNV pathogenicity
