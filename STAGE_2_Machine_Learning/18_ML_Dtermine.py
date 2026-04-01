"""
Machine Learning Pipeline for Lineage Prediction from Aligned Nucleotide Sequences

This script:
1. Loads aligned FASTA sequences and one-hot encodes them
2. Merges with lineage metadata
3. Removes zero-variance features
4. Compares multiple ML models using PyCaret with 10-fold CV
5. Extracts top 100 features from the best tree-based model

Required inputs:
- FASTA file with aligned nucleotide sequences
- CSV file with 'Accession' and 'Lineage' columns
"""

import pandas as pd
import numpy as np
from Bio import SeqIO
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


def load_fasta_sequences(fasta_file):
    """
    Load sequences from FASTA file.

    Args:
        fasta_file: Path to FASTA file

    Returns:
        Dictionary with accession as key and sequence as value
    """
    print(f"Loading FASTA file: {fasta_file}")
    sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences[record.id] = str(record.seq).upper()

    print(f"Loaded {len(sequences)} sequences")
    if sequences:
        seq_length = len(next(iter(sequences.values())))
        print(f"Sequence length: {seq_length} nucleotides")

    return sequences


def one_hot_encode_sequences(sequences):
    """
    One-hot encode nucleotide sequences.

    Each position is split into 4 columns (A, T, C, G).
    Standard nucleotides get 1 in matching column, 0 in others.
    Ambiguous nucleotides or gaps get 0 in all columns.

    Args:
        sequences: Dictionary of {accession: sequence}

    Returns:
        DataFrame with one-hot encoded sequences
    """
    print("\nOne-hot encoding sequences...")

    # Get sequence length
    seq_length = len(next(iter(sequences.values())))

    # Initialize data structure
    data = []
    accessions = []

    # Valid nucleotides
    valid_nucs = {'A', 'T', 'C', 'G'}

    for accession, sequence in sequences.items():
        accessions.append(accession)
        row = []

        # Process each position
        for pos_idx, nucleotide in enumerate(sequence, start=1):
            # Create encoding for this position
            encoding = {
                'A': 0,
                'T': 0,
                'C': 0,
                'G': 0
            }

            # Set 1 for matching nucleotide if valid
            if nucleotide in valid_nucs:
                encoding[nucleotide] = 1

            # Append in order: A, T, C, G
            row.extend([encoding['A'], encoding['T'], encoding['C'], encoding['G']])

        data.append(row)

    # Create column names
    columns = []
    for pos in range(1, seq_length + 1):
        pos_str = f"p{pos:05d}"  # 5-digit format
        columns.extend([f"{pos_str}_A", f"{pos_str}_T", f"{pos_str}_C", f"{pos_str}_G"])

    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)
    df.insert(0, 'Accession', accessions)

    print(f"Created one-hot encoded DataFrame: {df.shape[0]} samples × {df.shape[1]} columns")

    return df


def load_metadata(metadata_file):
    """
    Load metadata from CSV or TSV file and extract Accession and Lineage columns.
    Handles different encodings automatically.

    Args:
        metadata_file: Path to CSV or TSV file

    Returns:
        DataFrame with Accession and Lineage columns
    """
    print(f"\nLoading metadata from: {metadata_file}")

    # Determine separator based on file extension
    if metadata_file.endswith('.tsv') or metadata_file.endswith('.txt'):
        separator = '\t'
        file_type = "TSV"
    else:
        separator = ','
        file_type = "CSV"

    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
    df = None

    for encoding in encodings:
        try:
            df = pd.read_csv(metadata_file, sep=separator, encoding=encoding)
            print(f"Successfully loaded {file_type} with encoding: {encoding}")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            continue

    if df is None:
        raise ValueError(f"Could not read {file_type} file with any common encoding. Tried: {encodings}")

    # Check required columns
    if 'Accession' not in df.columns or 'Lineage' not in df.columns:
        raise ValueError(f"{file_type} file must contain 'Accession' and 'Lineage' columns")

    # Extract only needed columns
    metadata = df[['Accession', 'Lineage']].copy()

    print(f"Loaded metadata: {metadata.shape[0]} samples")
    print(f"Lineage distribution:\n{metadata['Lineage'].value_counts()}")

    return metadata


def merge_data(encoded_df, metadata_df):
    """
    Merge one-hot encoded data with metadata based on Accession.

    Args:
        encoded_df: One-hot encoded DataFrame
        metadata_df: Metadata DataFrame with Accession and Lineage

    Returns:
        Merged DataFrame
    """
    print("\nMerging encoded sequences with metadata...")

    merged = encoded_df.merge(metadata_df, on='Accession', how='inner')

    # Reorder columns: Accession, Lineage, then all features
    feature_cols = [col for col in merged.columns if col not in ['Accession', 'Lineage']]
    merged = merged[['Accession', 'Lineage'] + feature_cols]

    print(f"Merged DataFrame shape: {merged.shape[0]} samples × {merged.shape[1]} columns")

    return merged


def remove_zero_variance_features(df):
    """
    Remove features with zero variance (same value across all samples).

    Args:
        df: DataFrame with Accession, Lineage, and feature columns

    Returns:
        DataFrame with zero-variance features removed
    """
    print("\nRemoving zero-variance features...")

    # Get feature columns (exclude Accession and Lineage)
    feature_cols = [col for col in df.columns if col not in ['Accession', 'Lineage']]

    # Calculate variance for each feature
    variances = df[feature_cols].var()

    # Find zero-variance features
    zero_var_features = variances[variances == 0].index.tolist()

    print(f"Found {len(zero_var_features)} zero-variance features to remove")

    # Remove zero-variance features
    if zero_var_features:
        df = df.drop(columns=zero_var_features)
        print(f"DataFrame shape after removal: {df.shape[0]} samples × {df.shape[1]} columns")

    return df


def run_pycaret_comparison(df):
    """
    Run PyCaret model comparison with 10-fold cross-validation.

    Args:
        df: DataFrame with Accession, Lineage, and features

    Returns:
        Tuple of (results DataFrame, list of best models)
    """
    print("\n" + "=" * 80)
    print("Starting PyCaret Model Comparison")
    print("=" * 80)

    from pycaret.classification import setup, compare_models, pull

    # Setup PyCaret
    print("\nSetting up PyCaret experiment...")
    clf_setup = setup(
        data=df,
        target='Lineage',
        session_id=42,
        fold=10,  # 10-fold cross-validation
        ignore_features=['Accession'],  # Don't use Accession as feature
        verbose=False,
        html=False
    )

    print("\nComparing all available models with 10-fold cross-validation...")
    print("This may take a while depending on dataset size...\n")

    # Compare all models
    best_models = compare_models(
        include=None,  # Include all available models
        fold=10,
        sort='Accuracy',  # Sort by accuracy
        n_select=10  # Return top 10 models
    )

    # Get comparison results
    results = pull()

    print("\n" + "=" * 80)
    print("MODEL COMPARISON RESULTS (Ranked by Accuracy)")
    print("=" * 80)
    print(results.to_string())

    return results, best_models


def extract_top_tree_model_features(best_models, results, df):
    """
    Extract top 100 features from the best tree-based model based on gain importance.

    Tree-based models include: Random Forest, Gradient Boosting, Extra Trees,
    LightGBM, XGBoost, AdaBoost, Decision Tree, CatBoost

    Args:
        best_models: List of best models from compare_models
        results: Results DataFrame from compare_models
        df: DataFrame with features

    Returns:
        DataFrame with top 100 features and their importance scores
    """
    print("\n" + "=" * 80)
    print("EXTRACTING TOP 100 FEATURES FROM BEST TREE-BASED MODEL")
    print("=" * 80)

    # Define tree-based model identifiers
    tree_model_keywords = [
        'Random Forest', 'Gradient Boosting', 'Extra Trees',
        'Light Gradient', 'XGBoost', 'Ada Boost', 'Decision Tree',
        'CatBoost'
    ]

    # Find the best tree-based model from results
    best_tree_model = None
    best_tree_name = None

    # Iterate through results to find top tree-based model
    for idx, row in results.iterrows():
        model_name = row['Model']
        if any(keyword in model_name for keyword in tree_model_keywords):
            best_tree_name = model_name
            # Find corresponding model in best_models list
            if isinstance(best_models, list):
                model_idx = results.index.get_loc(idx)
                if model_idx < len(best_models):
                    best_tree_model = best_models[model_idx]
            else:
                best_tree_model = best_models
            break

    if best_tree_model is None:
        print("No tree-based model found in top models!")
        return None

    print(f"\nBest tree-based model: {best_tree_name}")
    print(f"Model class: {best_tree_model.__class__.__name__}")

    # Get model accuracy from results
    model_accuracy = results.loc[results['Model'] == best_tree_name, 'Accuracy'].values[0]
    print(f"Model accuracy: {model_accuracy:.4f}")

    # Get feature names (exclude Accession and Lineage)
    feature_cols = [col for col in df.columns if col not in ['Accession', 'Lineage']]

    # Extract feature importances
    if hasattr(best_tree_model, 'feature_importances_'):
        importances = best_tree_model.feature_importances_
    else:
        print("Warning: Model does not have feature_importances_ attribute")
        return None

    # Create DataFrame with features and importance scores
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance_Gain': importances
    })

    # Sort by importance (descending) and get top 100
    importance_df = importance_df.sort_values('Importance_Gain', ascending=False)
    top_100 = importance_df.head(100).reset_index(drop=True)

    # Add rank column
    top_100.insert(0, 'Rank', range(1, len(top_100) + 1))

    print(f"\nTop 100 Features by Gain Importance:")
    print("=" * 80)
    print(top_100.to_string())

    # Summary statistics
    print("\n" + "=" * 80)
    print("IMPORTANCE STATISTICS")
    print("=" * 80)
    print(f"Model: {best_tree_name}")
    print(f"Total features: {len(feature_cols)}")
    print(f"Sum of all importances: {importances.sum():.6f}")
    print(f"Sum of top 100 importances: {top_100['Importance_Gain'].sum():.6f}")
    print(f"Percentage captured by top 100: {(top_100['Importance_Gain'].sum() / importances.sum() * 100):.2f}%")
    print(f"\nTop 10 most important features:")
    for idx, row in top_100.head(10).iterrows():
        print(f"  {row['Rank']}. {row['Feature']}: {row['Importance_Gain']:.6f}")

    return top_100


def main(fasta_file, metadata_file):
    """
    Main pipeline execution.

    Args:
        fasta_file: Path to FASTA file
        metadata_file: Path to CSV or TSV metadata file

    Returns:
        Tuple of (comparison results, top 100 features)
    """
    try:
        # Step 1: Load FASTA sequences
        sequences = load_fasta_sequences(fasta_file)

        # Step 2: One-hot encode sequences
        encoded_df = one_hot_encode_sequences(sequences)

        # Step 3: Load metadata
        metadata = load_metadata(metadata_file)

        # Step 4: Merge data
        merged_df = merge_data(encoded_df, metadata)

        # Step 5: Remove zero-variance features
        processed_df = remove_zero_variance_features(merged_df)

        # Step 6: Run PyCaret model comparison
        results, best_models = run_pycaret_comparison(processed_df)

        # Step 7: Extract top 100 features from best tree-based model
        top_features = extract_top_tree_model_features(best_models, results, processed_df)

        print("\n" + "=" * 80)
        print("Pipeline completed successfully!")
        print("=" * 80)

        return results, top_features

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise


if __name__ == "__main__":
    # INPUT FILES - MODIFY THESE PATHS
    FASTA_FILE = "file path /13_ML_WNVSeq.fasta"
    METADATA_FILE = "file path/08_Lineage_assignments.tsv"

    # Run pipeline
    results, top_features = main(FASTA_FILE, METADATA_FILE)