library(tidyverse)
library(seqinr)
library(stringr)

# -------------------------
# Paths
# -------------------------
tsv_path <- "file path /08_Lineage_assignments.tsv"
fasta_path <- "file path /03_cleaned_sequences.fasta"

out_L1 <- "file path /11_WNV_Lineage1.fasta"
out_L2 <- "file path /12_WNV_Lineage2.fasta"

# -------------------------
# Read TSV (authority)
# -------------------------
lineage_tsv <- read_tsv(tsv_path)

# Keep only Lineage_1 and Lineage_2 (safety)
lineage_tsv <- lineage_tsv %>%
  filter(Lineage %in% c("Lineage_1", "Lineage_2"))

# -------------------------
# Read FASTA
# -------------------------
fasta <- read.fasta(
  fasta_path,
  as.string = TRUE,
  seqtype = "DNA"
)

# -------------------------
# Extract accession from FASTA headers
# (handles >ACC or >ACC|something)
# -------------------------
fasta_accessions <- names(fasta) %>%
  str_extract("^[^|]+")

# -------------------------
# Build FASTA table
# -------------------------
fasta_df <- tibble(
  Accession = fasta_accessions,
  sequence  = fasta
)

# -------------------------
# Join FASTA with TSV
# ONLY keep matching accessions
# -------------------------
fasta_joined <- fasta_df %>%
  inner_join(lineage_tsv, by = "Accession")

# -------------------------
# Split by lineage
# -------------------------
fasta_L1 <- fasta_joined %>% filter(Lineage == "Lineage_1")
fasta_L2 <- fasta_joined %>% filter(Lineage == "Lineage_2")

# -------------------------
# Write FASTA files
# -------------------------
write.fasta(
  sequences = fasta_L1$sequence,
  names     = fasta_L1$Accession,
  file.out  = out_L1
)

write.fasta(
  sequences = fasta_L2$sequence,
  names     = fasta_L2$Accession,
  file.out  = out_L2
)

# -------------------------
# Reporting
# -------------------------
cat("\nFASTA sequences total:", length(fasta), "\n")
cat("Matched to TSV:", nrow(fasta_joined), "\n")
cat("Lineage 1 sequences:", nrow(fasta_L1), "\n")
cat("Lineage 2 sequences:", nrow(fasta_L2), "\n")
cat("Dropped sequences:", length(fasta) - nrow(fasta_joined), "\n")
