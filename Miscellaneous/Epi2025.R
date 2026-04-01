library(tidyverse)
library(sf)
library(tigris)
library(scales)

# Data
epi_data <- tibble(
  state = c("Connecticut","Maine","Massachusetts","New Hampshire","Rhode Island","Vermont",
            "New Jersey","New York","Pennsylvania","Illinois","Indiana","Michigan","Ohio",
            "Wisconsin","Iowa","Kansas","Minnesota","Missouri","Nebraska","North Dakota",
            "South Dakota","Delaware","Florida","Georgia","Maryland","North Carolina",
            "South Carolina","Virginia","West Virginia","Alabama","Kentucky","Mississippi",
            "Tennessee","Arkansas","Louisiana","Oklahoma","Texas","Arizona","Colorado",
            "Idaho","Montana","Nevada","New Mexico","Utah","Wyoming","Alaska",
            "California","Hawaii","Oregon","Washington"),
  rate_raw = c("6","1","9","NA","2","NA","22","59","79","149","39","50","45","27","56",
               "31","122","43","54","86","86","3","6","17","28","10","4","30","2","29",
               "38","37","18","25","70","55","127","65","285","9","16","2","51","48","15",
               "NA","112","NA","2","2")
) %>%
  mutate(
    rate = case_when(
      rate_raw == "<1" ~ 0.5,
      TRUE ~ as.numeric(rate_raw)
    ),
    rate_category = case_when(
      is.na(rate) ~ NA_character_,
      rate == 0 ~ "0",
      rate <= 9 ~ "1–9",
      rate <= 29 ~ "10–29",
      rate <= 79 ~ "30–79",
      TRUE ~ "80+"
    ),
    rate_category = factor(
      rate_category,
      levels = c("0","1–9","10–29","30–79","80+")
    )
  )

# Get state geometries (tigris; set resolution as needed)
options(tigris_use_cache = TRUE)
states_sf <- states(cb = TRUE, resolution = "20m") %>%
  filter(!STUSPS %in% c("PR","VI","GU","MP","AS")) %>%  # remove territories
  shift_geometry()  # shifts AK & HI to inset positions

# Join
map_data <- states_sf %>%
  left_join(epi_data, by = c("NAME" = "state"))

# Color palette (red burden theme)
cat_colors <- c(
  "0"      = "#FEE5D9",
  "1–9"    = "#FCBBA1",
  "10–29"  = "#FC9272",
  "30–79"  = "#FB6A4A",
  "80+"    = "#A50F15"
)

# Plot
ggplot(map_data) +
  geom_sf(aes(fill = rate_category), color = "white", linewidth = 0.3) +
  scale_fill_manual(
    values = cat_colors,
    name   = "Case report",
    na.value = "#EEEEEE",
    drop   = FALSE
  ) +
  guides(fill = guide_legend(
    title.position = "top",
    label.position = "right",
    keywidth  = unit(0.4, "cm"),
    keyheight = unit(0.4, "cm")
  )) +
  theme_void(base_size = 12) +
  theme(
    legend.position      = c(0.92, 0.25),
    legend.title         = element_text(size = 9, face = "bold"),
    legend.text          = element_text(size = 8),
    plot.title           = element_text(size = 13, face = "bold", hjust = 0),
    plot.subtitle        = element_text(size = 9, color = "grey40", hjust = 0),
    plot.caption         = element_text(size = 7, color = "grey55", hjust = 1),
    plot.margin          = margin(10, 10, 10, 10)
  ) +
  labs(
    title    = "West Nile virus case reported, by state - US, 2025",
    subtitle = "",
    caption  = "Source: CDC"
  )

# Save at publication resolution
ggsave("epi_density_map.pdf", width = 8, height = 5, dpi = 300)  # PDF for journals
ggsave("epi_density_map.png", width = 8, height = 5, dpi = 600)  # High-res PNG