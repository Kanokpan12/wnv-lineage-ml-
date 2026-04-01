library(Biostrings)
library(tidyverse)

# ---------- Step 1: File paths ----------
fasta_path <- "file path/13_ML_WNVSeq.fasta"
out_dir    <- "file path"  

# ---------- Step 2: Define gene coordinates ----------
genes <- list(
  E   = c(start = 871,  end = 2373),
  NS3 = c(start = 4516, end = 6372),
  NS5 = c(start = 7588, end = 10302)
)

# ---------- Step 3: Read FASTA ----------
seqs <- readDNAStringSet(fasta_path)

# ---------- Step 4: Extract genes and save ----------
for (gene in names(genes)) {
  
  nt_start <- genes[[gene]]["start"]
  nt_end   <- genes[[gene]]["end"]
  
  gene_seqs <- subseq(seqs, start = nt_start, end = nt_end)
  
  # Keep original FASTA names
  names(gene_seqs) <- names(seqs)
  
  # Save FASTA
  out_file <- file.path(out_dir, paste0(gene, "_gene.fasta"))
  writeXStringSet(gene_seqs, out_file)
  
  cat("Saved", gene, "gene sequences to", out_file, "(", length(gene_seqs), "sequences )\n")
}

cat("✅ All genes extracted and saved as separate FASTA files.\n")

#### SAVE SEPERATELY #####
library(Biostrings)
library(tidyverse)

# ---------- Paths ----------
fasta_files <- c(
  E   = "file path /E_gene.fasta",
  NS3 = "file path /NS3_gene.fasta",
  NS5 = "file path /NS5_gene.fasta"
)
lineage_path <- "file path/08_Lineage_assignments.tsv"
out_dir      <- "file path"

# ---------- Read TSV ----------
lineage_df <- read.delim(lineage_path, stringsAsFactors = FALSE)
lineage_df$Accession <- trimws(lineage_df$Accession)
lineage_df$Lineage   <- trimws(lineage_df$Lineage)

# ---------- Split each gene by lineage ----------
for (gene in names(fasta_files)) {
  
  # Read the cleaned gene FASTA
  seqs <- readDNAStringSet(fasta_files[gene])
  fasta_accessions <- names(seqs) %>% trimws()
  
  # Map lineage from TSV
  seq_lineage <- lineage_df$Lineage[match(fasta_accessions, lineage_df$Accession)]
  
  # Check for unmatched sequences
  unmatched <- fasta_accessions[is.na(seq_lineage)]
  if (length(unmatched) > 0) {
    cat("\nWarning: Some sequences in", gene, "FASTA not found in TSV:\n")
    print(unmatched)
  }
  
  # Loop over the two lineages
  for (lin in c("Lineage_1", "Lineage_2")) {
    idx <- which(seq_lineage == lin)
    if (length(idx) == 0) {
      cat("No sequences for", gene, lin, "\n")
      next
    }
    
    out_seqs <- seqs[idx]
    names(out_seqs) <- paste0(fasta_accessions[idx], "|Lineage=", lin, "|", gene)
    
    # Save the separate FASTA
    out_file <- file.path(out_dir, paste0(gene, "_", lin, ".fasta"))
    writeXStringSet(out_seqs, out_file)
    
    cat("Saved", gene, lin, "with", length(idx), "sequences to", out_file, "\n")
  }
}

cat("\n✅ All done! You now have 6 FASTA files (3 genes × 2 lineages).\n")

