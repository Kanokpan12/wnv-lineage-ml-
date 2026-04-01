library(tidyverse)

# ==============================
# 1️⃣ Read CSV and clean column names
# ==============================
clinical_data <- read.csv("C:\\Users\\ksriwong\\Desktop\\ClinicalScore.csv", check.names = FALSE)

# Replace any empty column names
colnames(clinical_data) <- ifelse(colnames(clinical_data) == "",
                                  paste0("V", seq_along(colnames(clinical_data))),
                                  colnames(clinical_data))

# ==============================
# 2️⃣ Add Day column
# ==============================
clinical_data <- clinical_data %>%
  mutate(Day = 0:(nrow(clinical_data)-1)) %>%
  filter(Day <= 21)  # Endpoint Day 21

# ==============================
# 3️⃣ Wide to long format
# ==============================
clinical_long <- clinical_data %>%
  pivot_longer(
    cols = -Day,
    names_to = "Mouse",
    values_to = "Score"
  ) %>%
  mutate(Group = case_when(
    str_detect(Mouse, "Mock") ~ "Mock",
    str_detect(Mouse, "WNV_TX") ~ "WNV_TX",
    str_detect(Mouse, "WNV_MAD") ~ "WNV_MAD",
    TRUE ~ NA_character_
  )) %>%
  drop_na(Score)  # remove NA from mice that died

# ==============================
# 4️⃣ Summarize mean ± SEM per day
# ==============================
clinical_summary <- clinical_long %>%
  group_by(Day, Group) %>%
  summarise(
    Mean = mean(Score, na.rm = TRUE),
    SEM  = sd(Score, na.rm = TRUE) / sqrt(sum(!is.na(Score))),
    .groups = "drop"
  )

# ==============================
# 5️⃣ Plot
# ==============================
ggplot(clinical_summary, aes(x = Day, y = Mean, color = Group, fill = Group)) +
  geom_line(size = 1.2) +
  geom_ribbon(aes(ymin = Mean - SEM, ymax = Mean + SEM), alpha = 0.2, color = NA) +
  scale_color_manual(values = c("Mock" = "#7F7F7F",
                                "WNV_TX" = "#E41A1C",
                                "WNV_MAD" = "#984EA3")) +
  scale_fill_manual(values = c("Mock" = "#BEBEBE",
                               "WNV_TX" = "#FC9272",
                               "WNV_MAD" = "#BC80BD")) +
  labs(
    x = "Days Post-Inoculation",
    y = "Mean Clinical Score",
    title = ""
  ) +
  theme_classic(base_size = 14) +
  theme(
    axis.title.x = element_text(face = "bold", size = 14),
    axis.title.y = element_text(face = "bold", size = 14),
    axis.text = element_text(size = 12),
    legend.position = "top",
    legend.title = element_blank()
  ) +
  scale_x_continuous(breaks = seq(0, 21, 3)) +
  scale_y_continuous(breaks = seq(0, 5, 0.5))