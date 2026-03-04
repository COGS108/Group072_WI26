import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def get_status(s):
    if pd.isna(s) or 'Unknown' in s: return None
    return 'Neutered/Spayed' if 'Spayed' in s or 'Neutered' in s else 'Intact'

def plot_neutering_impact(combined):
    combined['neutered_status'] = combined['sex_upon_outcome'].apply(get_status)
    plot_df = combined.dropna(subset=['neutered_status'])

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=plot_df, 
        x='outcome_animal_type', 
        y='length_of_stay', 
        hue='neutered_status',
        showfliers=False
    )

    plt.title('Impact of Neutering on Length of Stay (0-60 days)')
    plt.ylabel('Days in Shelter')
    plt.xlabel('Animal Type')
    plt.ylim(0, 60)
    plt.legend(title='Status')
    plt.show()

    summary = plot_df.groupby(['outcome_animal_type', 'neutered_status'])['length_of_stay'].median()
    print(summary)


def plot_animal_correlation(df, animal_type):
    # Filter by animal type
    df_filtered = df[df['outcome_animal_type'] == animal_type].copy()
    
    # Parse age to days
    def parse_age_to_days(age_str):
        if pd.isna(age_str):
            return np.nan
        parts = str(age_str).split()
        if len(parts) < 2:
            return np.nan
        try:
            val = int(parts[0])
            unit = parts[1].lower()
            if 'year' in unit:
                return val * 365
            elif 'month' in unit:
                return val * 30
            elif 'week' in unit:
                return val * 7
            elif 'day' in unit:
                return val
        except:
            return np.nan
        return np.nan

    df_filtered['age_in_days'] = df_filtered['age_upon_outcome'].apply(parse_age_to_days)
    
    # Extract fixed status (1 for fixed, 0 for intact)
    def parse_fixed_status(s):
        if pd.isna(s) or 'Unknown' in str(s): 
            return np.nan
        return 1 if 'Spayed' in str(s) or 'Neutered' in str(s) else 0

    df_filtered['is_fixed'] = df_filtered['sex_upon_outcome'].apply(parse_fixed_status)
    
    # Select columns for correlation
    corr_cols = ['age_in_days', 'is_fixed', 'length_of_stay']
    corr_df = df_filtered[corr_cols].dropna()
    
    if len(corr_df) < 2:
        print(f"Not enough data to plot correlation for {animal_type}.")
        return

    # Calculate correlation matrix
    corr_matrix = corr_df.corr()
    
    # Plotting
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True)
    plt.title(f'Correlation Matrix for {animal_type}s: Age, Fixed Status, and Length of Stay')
    plt.show()    
