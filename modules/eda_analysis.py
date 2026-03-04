import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def parse_age_to_days(age_str):
    if pd.isna(age_str):
        return np.nan
    parts = age_str.split()
    if len(parts) != 2:
        return np.nan
    val, unit = parts
    try:
        val = int(val)
    except ValueError:
        return np.nan
    if 'year' in unit:
        return val * 365
    elif 'month' in unit:
        return val * 30
    elif 'week' in unit:
        return val * 7
    elif 'day' in unit:
        return val
    return np.nan

def add_derived_features(df):
    """Adds spay_neuter_status and age_numeric features."""
    if 'spay_neuter_status' not in df.columns:
        df['spay_neuter_status'] = df['sex_upon_outcome'].apply(
            lambda x: 'Spayed/Neutered' if pd.notna(x) and ('Spayed' in x or 'Neutered' in x) 
            else ('Intact' if pd.notna(x) and 'Intact' in x else 'Unknown')
        )
    if 'age_numeric_days' not in df.columns:
        df['age_numeric_days'] = df['age_upon_outcome'].apply(parse_age_to_days)
        df['age_numeric_years'] = df['age_numeric_days'] / 365.0
    
    if 'is_adopted' not in df.columns:
        df['is_adopted'] = df['outcome_type'].apply(lambda x: 'Adopted' if pd.notna(x) and x == 'Adoption' else 'Not Adopted')
    # Numeric encodings for correlation
    df['is_adopted_numeric'] = df['is_adopted'].apply(lambda x: 1 if x == 'Adopted' else 0)
    df['is_spay_neuter_numeric'] = df['spay_neuter_status'].apply(lambda x: 1 if x == 'Spayed/Neutered' else 0)
    
    return df

def plot_animal_correlation(df, animal_type):
    """
    Plots a comprehensive view for the given animal type including:
    1. A correlation heatmap of key numerical variables
    2. A scatter plot of Age vs Length of Stay, colored by Adoption Status
    """
    df_animal = df[df['intake_animal_type'] == animal_type].copy()
    df_animal = add_derived_features(df_animal)
    
    # Filter highly extreme outliers for length of stay for better visualization
    q_hi = df_animal['length_of_stay'].quantile(0.95)
    df_animal = df_animal[df_animal['length_of_stay'] < q_hi]
    df_animal = df_animal.dropna(subset=['age_numeric_years'])
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Correlation Heatmap
    corr_cols = ['age_numeric_years', 'length_of_stay', 'is_adopted_numeric', 'is_spay_neuter_numeric']
    corr_matrix = df_animal[corr_cols].corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[0], vmin=-1, vmax=1)
    axes[0].set_title(f'{animal_type} - Correlation Matrix')
    axes[0].set_xticklabels(['Age (Years)', 'Length of Stay', 'Adopted (1/0)', 'Spayed/Neutered (1/0)'], rotation=45)
    axes[0].set_yticklabels(['Age (Years)', 'Length of Stay', 'Adopted (1/0)', 'Spayed/Neutered (1/0)'], rotation=0)

    # 2. Scatter / Trend plot
    # Sample data to avoid overplotting
    df_sample = df_animal.sample(min(3000, len(df_animal)), random_state=42)
    sns.scatterplot(data=df_sample, x='age_numeric_years', y='length_of_stay', hue='is_adopted', style='spay_neuter_status', alpha=0.6, ax=axes[1])
    axes[1].set_title(f'{animal_type} - Age vs Length of Stay Trends')
    axes[1].set_xlabel('Age (Years)')
    axes[1].set_ylabel('Length of Stay (Days)')
    
    plt.tight_layout()
    plt.show()
