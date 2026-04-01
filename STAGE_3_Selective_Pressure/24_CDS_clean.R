library(seqinr)

# ---------- List of 6 input FASTA files ----------
fasta_files <- c(
  "E_Lineage_1"   = "file path /E_Lineage_1.fasta",
  "E_Lineage_2"   = "file path /E_Lineage_2.fasta",
  "NS3_Lineage_1" = "file path /NS3_Lineage_1.fasta",
  "NS3_Lineage_2" = "file path /NS3_Lineage_2.fasta",
  "NS5_Lineage_1" = "file path/NS5_Lineage_1.fasta",
  "NS5_Lineage_2" = "file path /NS5_Lineage_2.fasta"
)

# ---------- Functions ----------
is_complete_sequence <- function(seq) {
  seq_char <- as.character(seq)
  
  if (any(seq_char == "-")) return(list(complete = FALSE, reason = "contains gaps"))
  if (any(seq_char[-length(seq_char)] == "X")) return(list(complete = FALSE, reason = "contains undetermined amino acids (X)"))
  if (any(seq_char[-length(seq_char)] == "*")) return(list(complete = FALSE, reason = "contains premature stop codon"))
  if (any(seq_char %in% c("B","Z","J"))) return(list(complete = FALSE, reason = "contains ambiguous amino acids"))
  
  return(list(complete = TRUE, reason = ""))
}

remove_last_aa <- function(seq) {
  seq_char <- as.character(seq)
  if (length(seq_char) > 0) return(seq_char[-length(seq_char)])
  return(seq_char)
}

# ---------- Loop over all FASTA files ----------
for (name in names(fasta_files)) {
  input_file <- fasta_files[name]
  
  if (!file.exists(input_file)) {
    cat("File not found:", input_file, "\n")
    next
  }
  
  cat("\nProcessing file:", input_file, "\n")
  
  sequences <- read.fasta(input_file, seqtype = "AA", as.string = FALSE)
  cat("Total sequences read:", length(sequences), "\n")
  
  kept_seqs <- list()
  removed_seqs <- list()
  
  for (seq_name in names(sequences)) {
    seq <- sequences[[seq_name]]
    check <- is_complete_sequence(seq)
    
    if (!check$complete) {
      removed_seqs[[seq_name]] <- check$reason
      cat("REMOVED:", seq_name, "-", check$reason, "\n")
    } else {
      kept_seqs[[seq_name]] <- kept_seqs[[seq_name]]
    }
  }
  
  cat("Kept sequences:", length(kept_seqs), "Removed sequences:", length(removed_seqs), "\n")
  
  if (length(kept_seqs) > 0) {
    # Overwrite the original file
    write.fasta(sequences = kept_seqs, names = names(kept_seqs), 
                file.out = input_file, open = "w")
    cat("Original FASTA overwritten with cleaned sequences:", input_file, "\n")
  } else {
    cat("No sequences passed filters. Original file not modified.\n")
  }
}

cat("\n✅ All 6 FASTA files have been cleaned and overwritten.\n")

#### RANDOM PICK SAMPLE ######
library(Biostrings)
library(dplyr)

# -----------------------------
# 1. Parameters
# -----------------------------
genes <- c("E", "NS3", "NS5")
lineages <- c("Lineage_1", "Lineage_2")
mandatory_list <- list(
  "Lineage_1" = c("DQ176637", "KT934804"),
  "Lineage_2" = c("DQ176636")
)
folder_path <- "/Users/kanokpant.sriwong/Desktop"  # folder with FASTA files
set.seed(123)  # reproducibility

# -----------------------------
# 2. Function to extract accession
# -----------------------------
extract_accession <- function(fasta) {
  headers <- names(fasta)
  accession <- sapply(strsplit(headers, "\\|"), `[`, 1)  # first part of header
  data.frame(
    full_header = headers,
    accession = accession,
    stringsAsFactors = FALSE
  )
}

# -----------------------------
# 3. Collect all accessions per lineage
# -----------------------------
accessions_per_lineage <- list()

for (lg in lineages) {
  all_acc <- c()
  for (g in genes) {
    fasta_file <- file.path(folder_path, paste0(g, "_", lg, ".fasta"))
    if (!file.exists(fasta_file)) stop("File not found: ", fasta_file)
    
    fasta <- readDNAStringSet(fasta_file)
    df <- extract_accession(fasta)
    all_acc <- c(all_acc, df$accession)
  }
  all_acc <- unique(all_acc)
  
  # Mandatory + random
  mandatory <- mandatory_list[[lg]]
  n_random <- ifelse(lg == "Lineage_1", 98, 99)
  acc_pool <- setdiff(all_acc, mandatory)
  acc_random <- sample(acc_pool, n_random)
  
  accessions_per_lineage[[lg]] <- c(mandatory, acc_random)
}

# -----------------------------
# 4. Filter and overwrite FASTA files
# -----------------------------
for (lg in lineages) {
  for (g in genes) {
    fasta_file <- file.path(folder_path, paste0(g, "_", lg, ".fasta"))
    fasta <- readDNAStringSet(fasta_file)
    
    df <- extract_accession(fasta)
    
    # Keep only selected accessions
    acc_keep <- accessions_per_lineage[[lg]]
    keep_idx <- which(df$accession %in% acc_keep)
    fasta_selected <- fasta[keep_idx]
    
    # Clean headers
    names(fasta_selected) <- df$accession[keep_idx]
    
    # OVERWRITE original FASTA
    writeXStringSet(fasta_selected, fasta_file)
    cat("Overwritten:", fasta_file, "\n")
  }
}
