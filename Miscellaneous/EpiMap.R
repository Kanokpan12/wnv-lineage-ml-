library(tidyverse)
library(sf)
library(tigris)
library(viridis)

# Load file
epi_raw <- read.csv("C:/Users/ksriwong/Desktop/Epi.csv", check.names = FALSE)

#Step 1: clean data 
# Remove commas and <1, trim spaces
clean_number <- function(x){
  x <- gsub(",", "", x)
  x <- gsub("<1", "0.5", x)
  as.numeric(x)
}

epi_clean <- epi_raw %>%
  rename(state = `Region/State`) %>%
  mutate(state = trimws(state)) %>%
  
  # Keep only real states (remove regions & US)
  filter(!(state %in% c("United States","New England"))) %>%
  
  # Remove indentation spaces before state names
  mutate(state = gsub("^\\s+", "", state)) %>%
  
  # Clean numeric columns
  mutate(across(`2009`:`2018`, clean_number),
         Average = clean_number(Average),
         Median = clean_number(Median))

# Step 2: Map average case 
epi_map <- epi_clean %>%
  select(state, rate = Average)

# Step 3: Get US state shapfile 
states <- states(cb = TRUE, year = 2020, class = "sf") %>%
  mutate(state = NAME)

# Step 4: Join data to the map 
map_data <- states %>%
  left_join(epi_map, by = "state")

# Step 5: Plot distribution map 
ggplot(map_data) +
  geom_sf(aes(fill = rate), color = "white", size = 0.2) +
  scale_fill_viridis(option = "magma",
                     na.value = "grey90",
                     name = "Rate") +
  theme_void() +
  labs(title = "Rate by U.S. State")

map_data$group <- cut(map_data$rate,
                      breaks = 5)

ggplot(map_data) +
  geom_sf(aes(fill = group), color = "white", size = 0.2) +
  scale_fill_brewer(palette = "Reds", na.value = "grey90") +
  theme_void()