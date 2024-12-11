# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 23:30:52 2024

@author: nildi
"""

# Importació de llibreries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time 
import joblib

start_time = time.time()


# Carregar les dades des de l'arxiu CSV
df = pd.read_csv('./data/fantasy_data.csv')

# Comprovem les primeres files per assegurar-nos que les dades s'han carregat correctament
print(df.head())

# Carregar el DataFrame (df)
df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])

# Crear una nova columna per 'jornada' (a partir de 'kickoff_time')
df['round'] = df['kickoff_time'].dt.strftime('%Y-%m-%d')  # O una altra manera per identificar les jornades de manera única

# Ordenar les dades pel jugador i la jornada
df = df.sort_values(by=['player_id', 'round'])

# 1. Crear les característiques utilitzant les n darreres jornades
def create_features(df, n_prev_games):
    features = []
    target = []

    # Definim les columnes numèriques que utilitzarem
    numeric_cols = ['assists', 'bonus', 'bps', 'clean_sheets', 'creativity',
                   'goals_conceded', 'goals_scored', 'ict_index', 'influence', 
                   'minutes', 'own_goals', 'penalties_missed', 'penalties_saved', 
                   'red_cards', 'saves', 'selected', 'team_a_score', 'team_h_score', 
                   'threat', 'transfers_balance', 'transfers_in', 'transfers_out', 
                   'value', 'yellow_cards', 'ppm']

    # Definim les columnes categòriques
    categorical_cols = ['player_name', 'team', 'opponent_team']

    for player_id in df['player_id'].unique():
        player_data = df[df['player_id'] == player_id]

        for i in range(n_prev_games, len(player_data)):
            last_games = player_data.iloc[i-n_prev_games:i]
            current_game = player_data.iloc[i]
            
            # Obtenim les característiques numèriques dels últims partits
            X_numeric = last_games[numeric_cols].values.flatten()
            
            # Obtenim les característiques del proper partit
            additional_features = [
                player_id,  # int64
                current_game['player_name'],  # object
                current_game['team'],  # object
                current_game['opponent_team'],  # object
                1 if current_game['was_home'] else 0  # convertim bool a int
            ]
            
            # Combinem les característiques
            X = np.concatenate([X_numeric, additional_features])
            target_value = current_game['total_points']
            
            features.append(X)
            target.append(target_value)

    # Convertim a arrays numpy
    features = np.array(features, dtype=object)  # utilitzem dtype=object per poder mesclar tipus
    target = np.array(target)

    # Creem el DataFrame
    n_games = len(numeric_cols)
    feature_names = []
    
    # Afegim noms per cada característica de cada partit anterior
    for i in range(n_prev_games):
        for col in numeric_cols:
            feature_names.append(f"{col}_game_{i+1}")
    
    # Afegim noms per les característiques adicionals
    feature_names.extend(['player_id', 'player_name', 'team', 'opponent_team', 'was_home'])
    
    X_df = pd.DataFrame(features, columns=feature_names)

    # Convertim les variables categòriques en dummies
    X_df = pd.get_dummies(X_df, columns=['player_name', 'team', 'opponent_team'])

    return X_df, target



# 2. Crear les característiques i la variable dependent (puntuació)
n_prev_games = 5
X, y = create_features(df, n_prev_games)

# Convertir les dades a un DataFrame per facilitar les operacions
X_df = pd.DataFrame(X)

# Convertir les variables categòriques en dummies (one-hot encoding)
X_df = pd.get_dummies(X_df, columns=[X_df.columns[-5], X_df.columns[-4], X_df.columns[-3], X_df.columns[-2], X_df.columns[-1]], drop_first=True)

# 4. Dividir les dades en entrenament i prova
X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)


print('Començant entrenament...')
# 5. Entrenar el model de Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 6. Predicció de les puntuacions
y_pred = rf.predict(X_test)

# 7. Avaluar el model
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')


# Guardar el model i les columnes
print("Guardant el model i les columnes...")
joblib.dump(rf, 'saved_model/model_rf.joblib')
joblib.dump(X_df.columns, 'saved_model/model_columns.joblib')

# Distribució dels errors: Pots visualitzar com es distribueixen els errors entre les prediccions i els valors reals:
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Punts reals')
plt.ylabel('Punts predits')
plt.title('Comparació entre punts reals i predits')
plt.show()

# Importància de les característiques: Analitzar quines característiques tenen més impacte en les prediccions:
importances = rf.feature_importances_
sorted_indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[sorted_indices])
plt.xticks(range(len(importances)), X_df.columns[sorted_indices], rotation=90)
plt.title('Importància de les característiques')
plt.show()
# Mostrar temps d'execució
end_time = time.time()
print(f"Temps total d'execució: {end_time - start_time:.2f} segons")
