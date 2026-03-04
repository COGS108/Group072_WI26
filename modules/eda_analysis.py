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
