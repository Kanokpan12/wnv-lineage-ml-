########## ML model plot ##########

library(ggplot2)
library(dplyr)
library(tidyr)

# ----------------------------
# 1. Data
# ----------------------------
model_df <- data.frame(
  Model = c("Logistic","KNN","Random Forest","Gradient Boosting","LightGBM",
            "Decision Tree","AdaBoost","QDA","Dummy"),
  Accuracy = c(1.0000,1.0000,1.0000,1.0000,1.0000,0.9992,0.9977,0.8555,0.7969),
  AUC      = c(1.0000,1.0000,1.0000,1.0000,1.0000,0.9995,0.9986,1.0000,0.5000),
  Recall   = c(1.0000,1.0000,1.0000,1.0000,1.0000,0.9992,0.9977,0.8555,0.7969),
  Precision= c(1.0000,1.0000,1.0000,1.0000,1.0000,0.9993,0.9978,0.9383,0.6351)
)

# ----------------------------
# 2. Reshape to long format
# ----------------------------
plot_df <- model_df %>%
  pivot_longer(cols = c(Accuracy, AUC, Recall, Precision),
               names_to = "Metric", values_to = "Value")

# ----------------------------
# 3. Define pastel colors
# ----------------------------
colors <- c(
  "Accuracy" = "#4A90E2",  
  "AUC" = "#50C878",       
  "Recall" = "#F39C12",    
  "Precision" = "#E74C3C" 
)

# ----------------------------
# 4. Plot
# ----------------------------
ggplot(plot_df, aes(x = Model, y = Value, fill = Metric)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7, color = "black", linewidth = 0.3) +
  scale_fill_manual(values = colors) +
  scale_y_continuous(expand = c(0,0), limits = c(0,1.05)) +
  labs(
    x = "Model",
    y = "Score",
    title = "Comparison of Model Performance Metrics"
  ) +
  theme_bw() +
  theme(
    axis.title = element_text(face = "bold", size = 12),
    axis.text = element_text(face = "bold", size = 10),
    axis.text.x = element_text(angle = 0),
    plot.title = element_text(face = "bold", hjust = 0.5, size = 14),
    legend.title = element_text(face = "bold", size = 12),
    legend.text = element_text(size = 11),
    legend.position = "right",
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank()
  ) +
  coord_flip()

################################ the end ######################################


##################### NT and AA.csv and plot ####################################

library(dplyr)
library(tidyr)
library(stringr)

#Import csv
importance_df <- read.csv(
  "file path/Importance_gain.csv",
  stringsAsFactors = FALSE
)

#Protein ref
protein_ref <- tibble(
  start = c(1,124,291,792,1144,1375,1506,2125,2251,2274,2530),
  stop  = c(123,290,791,1143,1374,1505,2124,2250,2273,2529,3434),
  protein = c("C","prM","E","NS1","NS2A","NS2B","NS3","NS4A","2K","NS4B","NS5"),
  aa_length = c(123,167,501,352,231,131,619,126,23,255,905)
) %>%
  filter(protein != "2K")

#extraxt nt position from csv 
importance_parsed <- importance_df %>%
  separate(
    Feature,
    into = c("nt_pos", "allele"),
    sep = "_",
    remove = FALSE
  ) %>%
  mutate(
    nt_pos = as.integer(str_remove(nt_pos, "^p"))
  )

#convert nt to aa
importance_parsed <- importance_parsed %>%
  mutate(
    aa_pos = floor((nt_pos - 1) / 3) + 1
  )

#map aa --> pprotein + relative position
importance_mapped <- importance_parsed %>%
  rowwise() %>%
  mutate(
    protein = protein_ref$protein[
      aa_pos >= protein_ref$start &
        aa_pos <= protein_ref$stop
    ],
    protein_aa_pos = aa_pos - protein_ref$start[
      aa_pos >= protein_ref$start &
        aa_pos <= protein_ref$stop
    ] + 1
  ) %>%
  ungroup() %>%
  filter(!is.na(protein))

#final table
final_df <- importance_mapped %>%
  select(
    Rank,
    Feature,
    nt_pos,
    aa_pos,
    protein,
    protein_aa_pos,
    allele,
    Importance_Gain
  )
#output
write.csv(
  final_df,
  "file path/feature_importance_annotated_WNV.csv",
  row.names = FALSE
)

