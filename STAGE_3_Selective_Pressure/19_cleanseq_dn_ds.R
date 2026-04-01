#!/usr/bin/env Rscript

# Load required library
if (!require("seqinr", quietly = TRUE)) {
  install.packages("seqinr")
  library(seqinr)
}

# Function to check if sequence is complete
is_complete_sequence <- function(seq) {
  # Convert to character vector if not already
  seq_char <- as.character(seq)
  
  # Check for gaps (-)
  if (any(seq_char == "-")) {
    return(list(complete = FALSE, reason = "contains gaps"))
  }
  
  # Check for undetermined amino acids (X, but not at the end)
  # We'll check the whole sequence first, then handle terminal X separately
  if (any(seq_char[-length(seq_char)] == "X")) {
    return(list(complete = FALSE, reason = "contains undetermined amino acids (X)"))
  }
  
  # Check for premature stop codons (*)
  # Stop codon should only be at the end, if present
  if (any(seq_char[-length(seq_char)] == "*")) {
    return(list(complete = FALSE, reason = "contains premature stop codon"))
  }
  
  # Check for other ambiguous amino acids (B, Z, J)
  ambiguous <- c("B", "Z", "J")
  if (any(seq_char %in% ambiguous)) {
    return(list(complete = FALSE, reason = "contains ambiguous amino acids"))
  }
  
  return(list(complete = TRUE, reason = ""))
}

# Function to remove last amino acid
remove_last_aa <- function(seq) {
  seq_char <- as.character(seq)
  if (length(seq_char) > 0) {
    return(seq_char[-length(seq_char)])
  }
  return(seq_char)
}

# Main script
main <- function() {
  # Get command line arguments
  args <- commandArgs(trailingOnly = TRUE)
  
  if (length(args) < 2) {
    cat("Usage: Rscript filter_fasta.R <input_fasta> <output_fasta>\n")
    cat("Example: Rscript filter_fasta.R input.fasta output.fasta\n")
    quit(status = 1)
  }
  
  input_file <- args[1]
  output_file <- args[2]
  
  # Check if input file exists
  if (!file.exists(input_file)) {
    cat("Error: Input file", input_file, "does not exist.\n")
    quit(status = 1)
  }
  
  cat("Reading FASTA file:", input_file, "\n")
  
  # Read FASTA file
  sequences <- read.fasta(input_file, seqtype = "AA", as.string = FALSE)
  
  cat("Total sequences read:", length(sequences), "\n\n")
  
  # Track removed and kept sequences
  removed_seqs <- list()
  kept_seqs <- list()
  
  # Process each sequence
  for (seq_name in names(sequences)) {
    seq <- sequences[[seq_name]]
    
    # Check if sequence is complete
    check_result <- is_complete_sequence(seq)
    
    if (!check_result$complete) {
      removed_seqs[[seq_name]] <- check_result$reason
      cat("REMOVED:", seq_name, "-", check_result$reason, "\n")
    } else {
      # Remove last amino acid
      processed_seq <- remove_last_aa(seq)
      kept_seqs[[seq_name]] <- processed_seq
    }
  }
  
  # Report summary
  cat("\n=== SUMMARY ===\n")
  cat("Total sequences:", length(sequences), "\n")
  cat("Removed sequences:", length(removed_seqs), "\n")
  cat("Kept sequences:", length(kept_seqs), "\n")
  
  if (length(removed_seqs) > 0) {
    cat("\n=== REMOVED SEQUENCES ===\n")
    for (name in names(removed_seqs)) {
      cat(name, ":", removed_seqs[[name]], "\n")
    }
  }
  
  # Write output file if there are sequences to keep
  if (length(kept_seqs) > 0) {
    cat("\nWriting", length(kept_seqs), "sequences to:", output_file, "\n")
    write.fasta(sequences = kept_seqs, names = names(kept_seqs), 
                file.out = output_file, open = "w")
    cat("Done!\n")
  } else {
    cat("\nWarning: No sequences passed the filters. Output file not created.\n")
  }
}

# Run main function
main()