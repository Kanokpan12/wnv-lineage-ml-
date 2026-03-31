## ===============================
## Load required packages
## ===============================
library(ape)
library(ggtree)
library(ggplot2)
library(tidyverse)

## ===============================
## Input files
## ===============================
# Metadata
d <- read.csv("file path/00_WNV_information.csv",
              header = TRUE,
              stringsAsFactors = FALSE)

tree <- read.tree("file path/03_cleaned_sequences.fasta.contree.tree")

# Distance matrix from IQ-TREE
dist_mat <- read.table("file path/03_cleaned_sequences.fasta.mldist", format = "phylip")

## ===============================
## Continent mapping
## ===============================
continent_map <- list(
  Asia = c("Pakistan", "India", "Israel", "Turkey", "Iran", "China", "Japan", "South Korea",
           "Vietnam", "Thailand", "Indonesia", "Bangladesh", "Philippines", "Malaysia",
           "Singapore", "Sri Lanka", "Nepal", "Afghanistan", "Saudi Arabia", "Iraq",
           "Jordan", "Kuwait", "Qatar", "United Arab Emirates", "Oman", "Yemen",
           "Kazakhstan", "Uzbekistan", "Turkmenistan", "Tajikistan", "Kyrgyzstan",
           "Mongolia", "Laos", "Myanmar", "Cambodia", "Azerbaijan"),
  
  Africa = c("Uganda", "Senegal", "South Africa", "Kenya", "Nigeria", "Morocco",
             "Madagascar", "Egypt", "Algeria", "Ethiopia", "Zambia", "Zimbabwe",
             "Ghana", "Tanzania", "Congo", "Angola", "Cameroon", "Ivory Coast",
             "Sudan", "Rwanda", "Mozambique", "Malawi", "Botswana", "Liberia",
             "Burkina Faso", "Tunisia", "Namibia", "Sierra Leone", "Mauritius",
             "Chad", "Mali", "Democratic Republic of the Congo",
             "Central African Republic"),
  
  North_America = c("USA", "Canada", "Mexico", "Panama", "Costa Rica", "Honduras",
                    "Guatemala", "Belize", "El Salvador", "Nicaragua", "Cuba",
                    "Dominican Republic", "Jamaica", "Trinidad and Tobago",
                    "British Virgin Islands"),
  
  South_America = c("Brazil", "Argentina", "Chile", "Peru", "Venezuela", "Ecuador",
                    "Bolivia", "Paraguay", "Uruguay", "Guyana", "Suriname",
                    "French Guiana"),
  
  Europe = c("Spain", "Italy", "France", "Germany", "Russia", "Switzerland",
             "Netherlands", "Serbia", "Belgium", "Romania", "Greece", "Austria",
             "Hungary", "Poland", "Cyprus", "Portugal", "Sweden", "Finland",
             "Norway", "Denmark", "Ireland", "Czech Republic", "Slovakia",
             "Ukraine", "Slovenia", "Kosovo", "Bulgaria"),
  
  Australia = c("Australia", "New Zealand")
)

assign_continent <- function(country) {
  for (continent in names(continent_map)) {
    if (country %in% continent_map[[continent]]) {
      return(continent)
    }
  }
  return(NA)
}

d$continent <- sapply(d$Country, assign_continent)

## ===============================
## Match metadata to tree tips
## ===============================
tree_tips <- tree$tip.label

d_lin <- d %>%
  filter(Accession %in% tree_tips) %>%
  select(Accession, continent)

rownames(d_lin) <- d_lin$Accession
d_lin_continent <- d_lin["continent"]

## ===============================
## Colors
## ===============================
colcont <- c(
  North_America = "#a6cee3",
  South_America = "#1f78b4",
  Europe        = "#b2df8a",
  Africa        = "#33a02c",
  Asia          = "#fb9a99",
  Australia     = "#ff7f00"
)

## ===============================
## Plot tree + continent heatmap
## ===============================
p <- ggtree(tree, layout='circular', branch.length="none")

p_continent <- ggtree::gheatmap(
  p,
  d_lin_continent,
  offset = 0.01,
  width = 0.15,
  color = NA,
  colnames_angle = 90
) +
  scale_fill_manual(
    name = "Continent",
    values = colcont,
    na.value = "grey90"
  ) +
  theme(
    legend.title = element_text(face = "bold"),
    legend.text  = element_text(face = "bold")
  )

p_continent
