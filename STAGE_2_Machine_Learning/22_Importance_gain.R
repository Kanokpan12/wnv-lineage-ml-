library(ggplot2)
library(gggenes)
library(patchwork)  # for combining plots

# 1. Importance bar plot
p1 <- ggplot(df, aes(x = aa_pos, y = Importance_Gain)) +
  geom_col(fill = "red") +
  labs(
    x = NULL,  # Remove x-axis label since it'll be shared with bottom plot
    y = "Sum Importance Gain"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_blank(),  # Hide x-axis text
    axis.ticks.x = element_blank()  # Hide x-axis ticks
  )

# 2. Genome annotation with gggenes
p2 <- ggplot(genes_ref, aes(xmin = start, xmax = end, y = molecule, fill = gene, label = gene)) +
  geom_gene_arrow() +
  geom_gene_label(size = 8, fontface = "bold") +  # Add gene labels inside the arrows
  facet_wrap(~ molecule, scales = "free", ncol = 1) +
  scale_fill_brewer(palette = "Set3") +
  labs(
    x = "Amino Acid Position",
    y = NULL
  ) +
  theme_genes() +
  theme(
    legend.position = "none"  
  )

# 3. Combine plots vertically
combined_plot <- p1 / p2 + 
  plot_layout(heights = c(2, 0.1))  # Adjust relative heights as needed

# Display
combined_plot

# 1. Sum Importance_Gain by gene (protein)
gene_summary <- df %>%
  group_by(protein) %>%
  summarise(
    Total_Importance = sum(Importance_Gain, na.rm = TRUE),
    min_pos = min(aa_pos, na.rm = TRUE),  # Get min position for ordering
    .groups = "drop"
  ) %>%
  arrange(min_pos) %>%  # Order by genomic position
  mutate(protein = factor(protein, levels = protein))  # Fix order for plotting

# 2. Create gene summary bar plot
p_gene_summary <- ggplot(gene_summary, aes(x = protein, y = Total_Importance)) +
  geom_col(aes(fill = protein), color = 'black', linewidth = 0.5) +
  scale_fill_brewer(palette = "Set3") +
  labs(
    x = "",
    y = "Total Importance Gain"
  ) +
  theme_minimal() +
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 45, hjust = 1, face = "bold"), 
    axis.text.y = element_text(face ="bold"), 
    axis.title.x = element_text(face ="bold"),
    axis.title.y = element_text(face ="bold"),
  )

p_gene_summary

