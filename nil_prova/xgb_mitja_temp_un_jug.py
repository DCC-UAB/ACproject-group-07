# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 16:40:16 2024

@author: nildi
"""

# Importació de llibreries
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb

# Carregar les dades des de l'arxiu CSV
df = pd.read_csv('./data/fantasy_data.csv')

# Comprovem les primeres files per assegurar-nos que les dades s'han carregat correctament
print(df.head())

# Carregar el DataFrame (df)
df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])

# Crear una nova columna per 'jornada' (a partir de 'kickoff_time')
df['round'] = df['kickoff_time'].dt.strftime('%Y-%m-%d')

# Afegir una columna de número de jornada basada en l'ordre per jugador
def assign_round_numbers(df):
    df['round_number'] = df.groupby('player_id').cumcount() + 1
    return df

df = assign_round_numbers(df)

# Ordenar les dades pel jugador i la jornada
df = df.sort_values(by=['player_id', 'round'])

# Dividir en train i test segons la meitat de les jornades per jugador
def split_train_test_half(df):
    train_data = []
    test_data = []

    for player_id, player_data in df.groupby('player_id'):
        num_rounds = len(player_data)
        split_index = (num_rounds + 1) // 2  # La meitat, afegint una extra al train si imparell
        train_data.append(player_data.iloc[:split_index])  # Primera meitat (o una extra si imparell)
        test_data.append(player_data.iloc[split_index:])   # Segona meitat

    train_df = pd.concat(train_data)
    test_df = pd.concat(test_data)

    return train_df, test_df

# Dividim el dataset
train_df, test_df = split_train_test_half(df)

# Característiques i target
feature_columns = [
    'assists', 'bonus', 'bps', 'clean_sheets', 'creativity',
    'goals_conceded', 'goals_scored', 'ict_index', 'influence',
    'minutes', 'own_goals','red_cards', 'saves', 'selected', 'team_a_score', 'team_h_score',
    'threat', 'transfers_balance', 'transfers_in', 'transfers_out',
    'value', 'was_home'
]

X_train = train_df[feature_columns]
y_train = train_df['total_points']

X_test = test_df[feature_columns]
y_test = test_df['total_points']

# Entrenar el model XGBoost
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)

# Mostrar la llista de jugadors disponibles
print("Jugadors disponibles:")
for player_name in df['player_name'].unique():
    print(player_name)

# Funcionalitat per predir punts d'un jugador concret en una jornada
while True:
    player_name = input("Introdueix el nom del jugador (o 'q' per sortir): ")
    if player_name.lower() == 'q':
        break

    try:
        # Seleccionar dades del jugador
        player_data = df[df['player_name'].str.contains(player_name, case=False, na=False)]
        player_test_data = test_df[test_df['player_name'].str.contains(player_name, case=False, na=False)]

        if player_data.empty:
            print("No s'ha trobat dades per al jugador especificat.")
            continue

        if player_test_data.empty:
            print("Aquest jugador no té jornades en el conjunt de test.")
            continue

        # Mostrar les jornades disponibles en el conjunt de test
        available_rounds = player_test_data[['round_number', 'round']].drop_duplicates()
        print("Jornades disponibles en el test:")
        for _, row in available_rounds.iterrows():
            print(f"Jornada {row['round_number']}: {row['round']}")

        round_number = int(input("Introdueix el número de la jornada: "))
        if round_number not in available_rounds['round_number'].values:
            print("La jornada introduïda no està disponible per aquest jugador en el test.")
            continue

        # Seleccionar les característiques del jugador per a la jornada especificada
        X_player = player_test_data[player_test_data['round_number'] == round_number][feature_columns]
        predicted_points = model.predict(X_player)

        print(f"Predicció per al jugador {player_name} en la jornada {round_number}: {predicted_points[0]:.2f} punts.")

    except Exception as e:
        print(f"Error: {e}")
