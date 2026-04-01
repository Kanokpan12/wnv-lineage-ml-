library(Biostrings)
library(dplyr)
library(tidyr)
library(ggplot2)
library(ggrepel)

# Load alignment and lineage data
alignment <- readAAStringSet("file path: 14_AA_AllSeq.fasta")
lineage_df <- read.table("file path: 08_Lineage_assignments.tsv", header = TRUE, sep = "\t")
p <- read.table("file path: 15_start_stop_proteinRef.tsv", header = TRUE, sep = "\t") %>%
  filter(protein != "2K")

# Convert alignment to character matrix
alignment_matrix <- do.call(rbind, strsplit(as.character(alignment), split = ""))
rownames(alignment_matrix) <- names(alignment)
alignment_df <- as.data.frame(alignment_matrix, stringsAsFactors = FALSE) %>%
  tibble::rownames_to_column(var = "Accession")

# Rename AA columns to V1, V2, ...
colnames(alignment_df)[-1] <- paste0("V", seq_len(ncol(alignment_df) - 1))

# Merge alignment with lineage info
merged_df <- alignment_df %>%
  left_join(lineage_df, by = "Accession")

# Define the position columns (assumes position columns are named like "1", "2", ...)
positions <- grep("^V[0-9]+$", names(merged_df), value = TRUE)

# Reshape to long format
plot_df <- merged_df[, c("Accession", "Lineage", positions)]
long_df <- plot_df %>%
  pivot_longer(cols = all_of(positions), names_to = "position", values_to = "amino_acid") %>%
  mutate(position = as.integer(gsub("^V", "", position)))  # Remove any non-numeric characters

# Map positions to protein name and relative position
position_to_protein <- data.frame(position = 1:max(p$stop), protein = NA, rel_pos = NA)
for (i in 1:nrow(p)) {
  idx <- p$start[i]:p$stop[i]
  position_to_protein$protein[idx] <- p$protein[i]
  position_to_protein$rel_pos[idx] <- 1:length(idx)
}

# Join the mapping
long_df <- long_df %>%
  left_join(position_to_protein, by = "position") %>%
  filter(!is.na(protein), !is.na(amino_acid)) %>%
  mutate(
    protein = factor(protein, levels = p$protein),
    rel_pos = factor(rel_pos, levels = 1:max(position_to_protein$rel_pos, na.rm = TRUE))
  )

# Shannon entropy calculation
entropy_df <- long_df %>%
  group_by(position, amino_acid) %>%
  summarise(count = n(), .groups = "drop") %>%
  group_by(position) %>%
  mutate(freq = count / sum(count)) %>%
  ungroup() %>%
  mutate(entropy_component = -freq * log2(freq)) %>%
  group_by(position) %>%
  summarise(entropy = sum(entropy_component, na.rm = TRUE), .groups = "drop") %>%
  left_join(position_to_protein, by = "position") %>%
  filter(!is.na(protein))

# Normalize entropy
entropy_df$norm_entropy <- entropy_df$entropy / log2(20)
entropy_df$protein <- factor(entropy_df$protein, levels = c("C", "prM", "E", "NS1", "NS2A", "NS2B", "NS3", "NS4A", "NS4B", "NS5"))

## Mean Shannon entropy per protein position 
mean_entropy_df <- entropy_df %>%
  group_by(protein, rel_pos) %>%
  summarise(
    mean_norm_entropy = mean(norm_entropy, na.rm = TRUE),
    .groups = "drop"
  )

## Summarize mean Shannon entropy per protein
mean_entropy_protein <- entropy_df %>%
  group_by(protein) %>%
  summarise(
    mean_norm_entropy = mean(norm_entropy, na.rm = TRUE),
    .groups = "drop"
  )

## Faceted plot with bold labels polyprotein
ggplot(mean_entropy_df, aes(x = as.numeric(rel_pos), y = mean_norm_entropy)) +
  geom_line(color = "steelblue", linewidth = 0.7) +
  facet_wrap(~ protein, scales = "free_x", ncol = 2) +
  labs(
    x = "Relative amino acid position",
    y = "Mean normalized Shannon entropy",
    title = "Mean Shannon entropy by WNV protein"
  ) +
  theme_bw() +
  theme(
    axis.title = element_text(face = "bold"),
    axis.text  = element_text(face = "bold"),
    strip.text = element_text(face = "bold", size = 12),
    plot.title = element_text(face = "bold", hjust = 0.5)
  )

## Sort proteins by mean entropy to see ranking
mean_entropy_protein_sorted <- mean_entropy_protein %>%
  arrange(desc(mean_norm_entropy))

print(mean_entropy_protein_sorted)

## Prepare data for side-by-side plot
comparison_df <- entropy_df %>%
  group_by(protein) %>%
  summarise(
    mean_entropy = mean(norm_entropy, na.rm = TRUE),
    total_entropy = sum(norm_entropy, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  pivot_longer(cols = c(mean_entropy, total_entropy),
               names_to = "metric", values_to = "value") %>%
  mutate(metric = factor(metric, 
                         levels = c("mean_entropy", "total_entropy"),
                         labels = c("Mean entropy\n(per position)", 
                                    "Total entropy\n(sum)")))

## Plot side-by-side
ggplot(comparison_df, aes(x = reorder(protein, value), y = value, fill = protein)) +
  geom_col(color = "black", width = 0.7) +
  facet_wrap(~ metric, scales = "free_x") +
  coord_flip() +
  scale_fill_brewer(palette = "Set3") +
  labs(
    x = "Protein",
    y = "Shannon entropy",
    title = "WNV protein variability: mean vs. total entropy"
  ) +
  theme_bw() +
  theme(
    axis.title = element_text(face = "bold", size = 12),
    axis.text = element_text(face = "bold", size = 10),
    plot.title = element_text(face = "bold", hjust = 0.5, size = 14),
    legend.position = "none",
    strip.text = element_text(face = "bold", size = 11)
  )

## Total entropy per protein (sum instead of mean)
total_entropy_protein <- entropy_df %>%
  group_by(protein) %>%
  summarise(
    total_norm_entropy = sum(norm_entropy, na.rm = TRUE),
    protein_length = n(),
    .groups = "drop"
  ) %>%
  arrange(desc(total_norm_entropy))

## Plot total entropy
ggplot(total_entropy_protein, aes(x = reorder(protein, total_norm_entropy), 
                                  y = total_norm_entropy, 
                                  fill = total_norm_entropy)) +
  geom_col(color = "black", width = 0.7) +
  geom_text(aes(label = round(total_norm_entropy, 1)), 
            hjust = -0.2, fontface = "bold", size = 3.5) +
  scale_fill_gradient(low = "lightblue", high = "darkred") +
  coord_flip() +
  labs(
    x = "Protein",
    y = "Total normalized Shannon entropy",
    title = "Total sequence variability across WNV proteins"
  ) +
  theme_bw() +
  theme(
    axis.title = element_text(face = "bold", size = 12),
    axis.text = element_text(face = "bold", size = 11),
    plot.title = element_text(face = "bold", hjust = 0.5, size = 14),
    legend.position = "none"
  )