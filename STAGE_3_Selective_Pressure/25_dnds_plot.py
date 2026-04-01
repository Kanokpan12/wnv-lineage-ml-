import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('file path')

# Auto-detect column names
def find_column(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None

position_col = find_column(df, ['site', 'Site', 'position', 'Position', 'codon', 'Codon', 'aa_pos'])
alpha_col = find_column(df, ['alpha', 'Alpha', 'dS', 'syn'])
beta_col = find_column(df, ['beta', 'Beta', 'dN', 'nonsyn'])
pvalue_col = find_column(df, ['pvalue', 'p-value', 'P-value', 'p_value', 'pValue', 'Pvalue'])

print(f"Detected columns:")
print(f"Position: {position_col}")
print(f"Alpha: {alpha_col}")
print(f"Beta: {beta_col}")
print(f"P-value: {pvalue_col}")

# Check if we found all required columns
if not all([position_col, alpha_col, beta_col, pvalue_col]):
    print("\n⚠️  Could not find all required columns!")
    exit()

# Create significance categories
df['significant'] = df[pvalue_col] < 0.05
df['selection'] = 'Neutral'
df.loc[(df[pvalue_col] < 0.05) & (df[beta_col] > df[alpha_col]), 'selection'] = 'Positive'
df.loc[(df[pvalue_col] < 0.05) & (df[beta_col] < df[alpha_col]), 'selection'] = 'Negative'

# Updated domain boundaries: MTase (1-300), RdRp (301-905)
df['domain'] = df[position_col].apply(lambda x: 'MTase' if x <= 300 else 'RdRp')

# Create the figure
fig, ax = plt.subplots(figsize=(18, 8))

# Set background
ax.set_facecolor('#fafafa')
fig.patch.set_facecolor('white')

# Add domain background shading
ax.axvspan(1, 300, alpha=0.15, color='#FFA500', zorder=0)
ax.axvspan(301, 905, alpha=0.15, color='#4169E1', zorder=0)

# Add vertical line at domain boundary
ax.axvline(x=300.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, zorder=1)

# Plot ALL sites with beta upward and alpha downward
for _, row in df.iterrows():
    if row['selection'] == 'Positive':
        # Significant positive selection - thicker, more opaque
        # Beta up (red/pink)
        ax.plot([row[position_col], row[position_col]], [0, row[beta_col]],
                color='#E91E63', linewidth=2, alpha=0.9, zorder=3)
        # Alpha down (blue)
        ax.plot([row[position_col], row[position_col]], [0, -row[alpha_col]],
                color='#2196F3', linewidth=2, alpha=0.9, zorder=3)
    elif row['selection'] == 'Negative':
        # Significant negative selection - thicker, more opaque
        # Beta up (red/pink)
        ax.plot([row[position_col], row[position_col]], [0, row[beta_col]],
                color='#E91E63', linewidth=2, alpha=0.9, zorder=3)
        # Alpha down (blue)
        ax.plot([row[position_col], row[position_col]], [0, -row[alpha_col]],
                color='#2196F3', linewidth=2, alpha=0.9, zorder=3)
    else:
        # Neutral sites - thinner, more transparent
        # Beta up (light red/pink)
        ax.plot([row[position_col], row[position_col]], [0, row[beta_col]],
                color='#E91E63', linewidth=0.8, alpha=0.3, zorder=2)
        # Alpha down (light blue)
        ax.plot([row[position_col], row[position_col]], [0, -row[alpha_col]],
                color='#2196F3', linewidth=0.8, alpha=0.3, zorder=2)

# Add small dots at the tips for significant sites
significant_sites = df[df['significant']]
ax.scatter(significant_sites[position_col], significant_sites[beta_col],
           color='#E91E63', s=15, alpha=0.9, zorder=4)
ax.scatter(significant_sites[position_col], -significant_sites[alpha_col],
           color='#2196F3', s=15, alpha=0.9, zorder=4)

# Reference line at 0
ax.axhline(y=0, color='black', linewidth=1.5, zorder=5)

# Labels
ax.set_xlabel('Codon →', fontsize=14, fontweight='bold')
ax.set_ylabel('Rate estimate', fontsize=12, fontweight='bold')
ax.set_title('',
             fontsize=16, fontweight='bold', pad=20)

# Axis limits
ax.set_xlim(0, 920)
y_max = max(df[beta_col].max(), df[alpha_col].max()) * 1.15
ax.set_ylim(-y_max, y_max)

# Grid
ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
ax.grid(True, axis='x', alpha=0.2, linestyle='--', linewidth=0.5)

# Domain labels
ax.text(150, y_max * 0.92, 'MTase', fontsize=14, fontweight='bold',
        ha='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))
ax.text(603, y_max * 0.92, 'RdRp', fontsize=14, fontweight='bold',
        ha='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))

# Add text labels for beta and alpha
ax.text(0.02, 0.98, 'β (non-synonymous)', transform=ax.transAxes,
        fontsize=11, verticalalignment='top', color='#E91E63', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
ax.text(0.02, 0.02, 'α (synonymous)', transform=ax.transAxes,
        fontsize=11, verticalalignment='bottom', color='#2196F3', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

plt.tight_layout()

# Get counts for summary
positive_sites = df[df['selection'] == 'Positive']
negative_sites = df[df['selection'] == 'Negative']
neutral_sites = df[df['selection'] == 'Neutral']

# Print detailed summary statistics
print(f"\n{'=' * 60}")
print(f"SUMMARY STATISTICS - NS5 Lineage 2 (Full-length)")
print(f"{'=' * 60}")
print(f"\nTotal sites analyzed: {len(df)}")
print(f"Significant sites (p < 0.05): {len(significant_sites)} ({len(significant_sites) / len(df) * 100:.2f}%)")
print(f"  - Positive selection (β > α): {len(positive_sites)} ({len(positive_sites) / len(df) * 100:.2f}%)")
print(f"  - Negative selection (β < α): {len(negative_sites)} ({len(negative_sites) / len(df) * 100:.2f}%)")
print(f"Neutral sites (p ≥ 0.05): {len(neutral_sites)} ({len(neutral_sites) / len(df) * 100:.2f}%)")

print(f"\n{'=' * 60}")
print(f"BY DOMAIN")
print(f"{'=' * 60}")
domain_summary = df.groupby(['domain', 'selection']).size().unstack(fill_value=0)
domain_summary['Total'] = domain_summary.sum(axis=1)
print(domain_summary)

# Calculate percentages by domain
print(f"\n{'=' * 60}")
print(f"SELECTION PERCENTAGE BY DOMAIN")
print(f"{'=' * 60}")
for domain in ['MTase', 'RdRp']:
    domain_df = df[df['domain'] == domain]
    total = len(domain_df)
    sig = len(domain_df[domain_df['significant']])
    pos = len(domain_df[domain_df['selection'] == 'Positive'])
    neg = len(domain_df[domain_df['selection'] == 'Negative'])

    print(f"\n{domain} ({total} sites):")
    print(f"  Significant: {sig} ({sig / total * 100:.2f}%)")
    print(f"  Positive: {pos} ({pos / total * 100:.2f}%)")
    print(f"  Negative: {neg} ({neg / total * 100:.2f}%)")

# Top sites under positive selection
if len(positive_sites) > 0:
    print(f"\n{'=' * 60}")
    print(f"TOP 10 SITES UNDER POSITIVE SELECTION")
    print(f"{'=' * 60}")
    top_positive = positive_sites.nlargest(10, beta_col)[[position_col, 'domain', alpha_col, beta_col, pvalue_col]]
    top_positive.columns = ['Position', 'Domain', 'α (dS)', 'β (dN)', 'p-value']
    print(top_positive.to_string(index=False))

# Top sites under negative selection
if len(negative_sites) > 0:
    print(f"\n{'=' * 60}")
    print(f"TOP 10 SITES UNDER NEGATIVE SELECTION")
    print(f"{'=' * 60}")
    top_negative = negative_sites.nsmallest(10, beta_col)[[position_col, 'domain', alpha_col, beta_col, pvalue_col]]
    top_negative.columns = ['Position', 'Domain', 'α (dS)', 'β (dN)', 'p-value']
    print(top_negative.to_string(index=False))

plt.show()