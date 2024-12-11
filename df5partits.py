# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 18:13:45 2024

@author: nildi
"""

import pandas as pd

# Carregar el dataset
file_path = 'data/fantasy_data.csv'  # Actualitza amb el camí correcte si cal
df = pd.read_csv(file_path)

# Funció per crear el dataset amb les característiques i punts esperats
def create_dataset_with_expected_points(df, n_matches=5):
    """
    Creates a dataset where each row represents a player's aggregated features
    and their expected points based on the last n_matches.
    """
    # Sort the dataframe by player and time
    df = df.sort_values(['player_id', 'kickoff_time'])

    # Create rolling features for the last n_matches
    rolling_features = ['minutes', 'goals_scored', 'assists', 'clean_sheets', 'total_points']
    for feature in rolling_features:
        rolling_column = f'rolling_{n_matches}_{feature}'
        df[rolling_column] = df.groupby('player_id')[feature].transform(
            lambda x: x.rolling(n_matches, min_periods=1).mean()
        )

    # Filter only the most recent record for each player for feature aggregation
    recent_records = df.groupby('player_id').tail(1)

    # Selecting the feature columns for the new dataset
    feature_columns = [f'rolling_{n_matches}_{feature}' for feature in rolling_features]
    final_dataset = recent_records[['player_id', 'player_name'] + feature_columns]

    return final_dataset




# Aplicar la funció per crear el dataset
feature_dataset = create_dataset_with_expected_points(df, n_matches=5)

# Guardar el dataset
output_path = 'data/df_5partits.csv'
feature_dataset.to_csv(output_path, index=False)

output_path
