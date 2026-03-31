import pandas as pd
import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

# -------------------------------
# File paths
# -------------------------------
metadata_file = "file path /00_WNV_information.csv"
dist_file = "file path /03_cleaned_sequences.fasta.mldist"
output_dir = "file path /FinalIQTree"

# -------------------------------
# Continent colors (colcont from R)
# -------------------------------
continent_colors = {
    'North_America': "#a6cee3",
    'South_America': "#1f78b4",
    'Europe': "#b2df8a",
    'Africa': "#33a02c",
    'Asia': "#fb9a99",
    'Australia': "#ff7f00",
    'Unknown': "#D1D5DB"
}

# -------------------------------
# Continent map (converted from R)
# -------------------------------
continent_map = {
    "Asia": ["Pakistan", "India", "Israel", "Turkey", "Iran", "China", "Japan", "South Korea",
             "Vietnam", "Thailand", "Indonesia", "Bangladesh", "Philippines", "Malaysia",
             "Singapore", "Sri Lanka", "Nepal", "Afghanistan", "Saudi Arabia", "Iraq",
             "Jordan", "Kuwait", "Qatar", "United Arab Emirates", "Oman", "Yemen",
             "Kazakhstan", "Uzbekistan", "Turkmenistan", "Tajikistan", "Kyrgyzstan",
             "Mongolia", "Laos", "Myanmar", "Cambodia", "Azerbaijan"],
    "Africa": ["Uganda", "Senegal", "South Africa", "Kenya", "Nigeria", "Morocco",
               "Madagascar", "Egypt", "Algeria", "Ethiopia", "Zambia", "Zimbabwe",
               "Ghana", "Tanzania", "Congo", "Angola", "Cameroon", "Ivory Coast",
               "Sudan", "Rwanda", "Mozambique", "Malawi", "Botswana", "Liberia",
               "Burkina Faso", "Tunisia", "Namibia", "Sierra Leone", "Mauritius",
               "Chad", "Mali", "Democratic Republic of the Congo", "Central African Republic"],
    "North_America": ["USA", "Canada", "Mexico", "Panama", "Costa Rica", "Honduras",
                      "Guatemala", "Belize", "El Salvador", "Nicaragua", "Cuba",
                      "Dominican Republic", "Jamaica", "Trinidad and Tobago",
                      "British Virgin Islands"],
    "South_America": ["Brazil", "Argentina", "Chile", "Peru", "Venezuela", "Ecuador",
                      "Bolivia", "Paraguay", "Uruguay", "Guyana", "Suriname", "French Guiana"],
    "Europe": ["Spain", "Italy", "France", "Germany", "Russia", "Switzerland",
               "Netherlands", "Serbia", "Belgium", "Romania", "Greece", "Austria",
               "Hungary", "Poland", "Cyprus", "Portugal", "Sweden", "Finland",
               "Norway", "Denmark", "Ireland", "Czech Republic", "Slovakia",
               "Ukraine", "Slovenia", "Kosovo", "Bulgaria"],
    "Australia": ["Australia", "New Zealand"]
}

# Create reverse lookup for fast mapping
country_to_continent = {country: cont for cont, countries in continent_map.items() for country in countries}

# -------------------------------
# Read metadata
# -------------------------------
metadata = pd.read_csv(metadata_file)
metadata['Continent'] = metadata['Country'].apply(lambda c: country_to_continent.get(str(c).strip(), 'Unknown'))

# -------------------------------
# Read distance matrix (PHYLIP-like)
# -------------------------------
with open(dist_file, 'r') as f:
    lines = f.readlines()

n_samples = int(lines[0].strip())
sample_names = []
dist_values = []

for line in lines[1:]:
    parts = line.strip().split()
    sample_names.append(parts[0])
    dist_values.append([float(x) for x in parts[1:]])

dist_matrix = np.array(dist_values)

# -------------------------------
# Run PCoA (MDS)
# -------------------------------
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, n_jobs=-1)
coords = mds.fit_transform(dist_matrix)

# -------------------------------
# Variance explained (approx)
# -------------------------------
n = len(dist_matrix)
H = np.eye(n) - np.ones((n, n)) / n
B = -0.5 * H @ (dist_matrix ** 2) @ H
eigenvalues = np.sort(np.linalg.eigvalsh(B))[::-1]
var_explained = eigenvalues[:2] / np.sum(eigenvalues[eigenvalues > 0]) * 100

# -------------------------------
# Combine coordinates with metadata
# -------------------------------
pcoa_df = pd.DataFrame({
    'PCoA1': coords[:, 0],
    'PCoA2': coords[:, 1],
    'Sample': sample_names
})

pcoa_df = pcoa_df.merge(
    metadata[[metadata.columns[0], 'Continent']].rename(columns={metadata.columns[0]: 'Sample'}),
    on='Sample', how='left'
)
pcoa_df['Continent'].fillna('Unknown', inplace=True)

# -------------------------------
# Plot all samples (publication ready)
# -------------------------------
plt.figure(figsize=(12, 10))
for cont, color in continent_colors.items():
    s = pcoa_df[pcoa_df['Continent'] == cont]
    if not s.empty:
        plt.scatter(
            s['PCoA1'], s['PCoA2'],
            s=120, alpha=0.8, c=color,
            label=f"{cont} (n={len(s)})",
            edgecolors='white', linewidth=0.5
        )

plt.xlabel(f'PCoA1 ({var_explained[0]:.2f}%)', fontsize=14, fontweight='bold')
plt.ylabel(f'PCoA2 ({var_explained[1]:.2f}%)', fontsize=14, fontweight='bold')
plt.title('West Nile Virus - Genetic similarity plot', fontsize=16, fontweight='bold')
plt.legend(loc='best', fontsize=11, frameon=True, title='Continent', title_fontsize=12)
plt.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()

plt.show()

