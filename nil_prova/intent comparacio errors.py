# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 18:07:47 2024

@author: nildi
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Carregar les dades des de l'arxiu CSV
dataset_path = 'data/fantasy_data.csv'
df = pd.read_csv(dataset_path)

# Preprocessar les dades
df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])
df['round'] = df['kickoff_time'].dt.strftime('%Y-%m-%d')
df = df.sort_values(by=['player_id', 'round'])

# Funció per crear característiques amb rolling
def create_features_optimized(df, n_prev_games):
    rolling_features = ['assists', 'bonus', 'bps', 'clean_sheets', 'creativity', 'goals_conceded',
                        'goals_scored', 'ict_index', 'influence', 'minutes', 'own_goals',
                        'penalties_missed', 'penalties_saved', 'red_cards', 'saves', 'selected',
                        'team_a_score', 'team_h_score', 'threat', 'transfers_balance',
                        'transfers_in', 'transfers_out', 'value', 'was_home', 'yellow_cards']

    # Calcular les mitjanes mòbils
    rolling_df = df.groupby('player_id')[rolling_features].rolling(window=n_prev_games, min_periods=1).mean().reset_index()

    # Afegir el target (total_points)
    rolling_df['total_points'] = df['total_points'].values

    # Filtrar només les files vàlides (després de calcular les rolling)
    rolling_df = rolling_df[rolling_df['level_1'] >= n_prev_games].reset_index(drop=True)

    # Eliminar la columna `level_1` (índex intermedi)
    rolling_df = rolling_df.drop(columns=['level_1'])

    return rolling_df.drop(columns=['total_points']), rolling_df['total_points']

# Llistat de valors de n_prev_games
n_prev_games_values = [1, 5, 10, 15, 20, 25, 30, 35]

# Llista per emmagatzemar els MAE
mae_values = []

# Bucle per calcular el MAE per a cada n_prev_games
for n_prev_games in n_prev_games_values:
    print(f"Calculant per n_prev_games = {n_prev_games}...")
    X, y = create_features_optimized(df, n_prev_games)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mae_values.append(mae)

# Crear el gràfic
plt.figure(figsize=(10, 6))
plt.plot(n_prev_games_values, mae_values, marker='o', linestyle='-', color='b', label='MAE')
plt.xlabel('n_prev_games (Número de partits anteriors considerats)', fontsize=12)
plt.ylabel('Mean Absolute Error (MAE)', fontsize=12)
plt.title('Comparació del MAE amb diferents valors de n_prev_games', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()
