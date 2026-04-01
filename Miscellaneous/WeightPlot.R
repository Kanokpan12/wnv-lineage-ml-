library(tidyverse)

# ==============================
# 1️⃣ Read CSV and clean column names
# ==============================
weight_data <- read.csv("C:\\Users\\ksriwong\\Desktop\\Weight.csv", check.names = FALSE)

# Replace any empty column names
colnames(weight_data) <- ifelse(colnames(weight_data) == "", paste0("V", seq_along(colnames(weight_data))), colnames(weight_data))

# ==============================
# 2️⃣ Add Day column
# ==============================
weight_data <- weight_data %>%
  mutate(Day = 0:(nrow(weight_data)-1)) %>%
  filter(Day <= 21)  # Endpoint Day 21

# ==============================
# 3️⃣ Wide to long format
# ==============================
weight_long <- weight_data %>%
  pivot_longer(
    cols = -Day,
    names_to = "Mouse",
    values_to = "Weight"
  ) %>%
  mutate(Group = case_when(
    str_detect(Mouse, "Mock") ~ "Mock",
    str_detect(Mouse, "WNV_TX") ~ "WNV_TX",
    str_detect(Mouse, "WNV_MAD") ~ "WNV_MAD",
    TRUE ~ NA_character_
  )) %>%
  drop_na(Weight)  # Remove NA values

# ==============================
# 4️⃣ Calculate % weight change per mouse
# ==============================
weight_long <- weight_long %>%
  group_by(Mouse) %>%
  mutate(PercentChange = 100 * (Weight / Weight[1] - 1)) %>%
  ungroup()

# ==============================
# 5️⃣ Summarize mean ± SEM
# ==============================
weight_summary <- weight_long %>%
  group_by(Day, Group) %>%
  summarise(
    Mean = mean(PercentChange, na.rm = TRUE),
    SEM  = sd(PercentChange, na.rm = TRUE) / sqrt(sum(!is.na(PercentChange))),
    .groups = "drop"
  )

# ==============================
# 6️⃣ Plot
# ==============================
ggplot(weight_summary, aes(x = Day, y = Mean, color = Group, fill = Group)) +
  geom_line(size = 1.2) +
  geom_ribbon(aes(ymin = Mean - SEM, ymax = Mean + SEM), alpha = 0.2, color = NA) +
  scale_color_manual(values = c("Mock" = "#7F7F7F", "WNV_TX" = "#E41A1C", "WNV_MAD" = "#984EA3")) +
  scale_fill_manual(values = c("Mock" = "#BEBEBE", "WNV_TX" = "#FC9272", "WNV_MAD" = "#BC80BD")) +
  labs(
    x = "Days Post-Inoculation",
    y = "% Weight Change",
    title = ""
  ) +
  theme_classic(base_size = 14) +
  theme(
    axis.title.x = element_text(face = "bold", size = 14),
    axis.title.y = element_text(face = "bold", size = 14),
    axis.text = element_text(size = 12),
    legend.position = "top",
    legend.title = element_blank(),
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5)
  ) +
  scale_x_continuous(breaks = seq(0, 21, 3)) +
  scale_y_continuous(breaks = seq(-20, 20, 5))

# summary of % weight change by group
weight_long %>%
  group_by(Group) %>%
  summarise(
    MeanPercentChange = mean(PercentChange, na.rm = TRUE),
    SDPercentChange   = sd(PercentChange, na.rm = TRUE),
    MinPercentChange  = min(PercentChange, na.rm = TRUE),
    MaxPercentChange  = max(PercentChange, na.rm = TRUE),
    .groups = "drop"
  )