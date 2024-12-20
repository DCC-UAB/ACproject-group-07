# Importació de llibreries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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

    for player_id in df['player_id'].unique():
        player_data = df[df['player_id'] == player_id]

        for i in range(n_prev_games, len(player_data)):
            last_games = player_data.iloc[i-n_prev_games:i]
            X = last_games[['assists', 'bonus', 'bps', 'clean_sheets', 'creativity', 'goals_conceded',
                            'goals_scored', 'ict_index', 'influence', 'minutes', 
                            'own_goals', 'penalties_missed', 'penalties_saved', 'red_cards', 'saves', 
                            'selected', 'team_a_score', 'team_h_score', 'threat', 'transfers_balance', 
                            'transfers_in', 'transfers_out', 'value', 'was_home', 'yellow_cards']].mean(axis=0)
            
            # Afegir player_id, player_name i team a les característiques
            X['player_id'] = player_id
            X['player_name'] = player_data.iloc[i]['player_name']
            X['team'] = player_data.iloc[i]['team']
            
            target_value = player_data.iloc[i]['total_points']
            features.append(X)
            target.append(target_value)
    
    return pd.DataFrame(features), target

# 2. Crear les característiques i la variable dependent (puntuació)
n_prev_games = 5
X, y = create_features(df, n_prev_games)

# 3. Convertir les variables categòriques en dummies (incloent 'player_name' i 'team')
X = pd.get_dummies(X, columns=['player_name', 'team'], drop_first=True)

# 4. Dividir les dades en entrenament i prova
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Entrenar el model de Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 6. Predicció de les puntuacions
y_pred = rf.predict(X_test)

# Interacció amb l'usuari
option = input('Vols veure la puntuació d\'un jugador? (y/n): ')

if option.lower() == 'y':
    player_name_input = input('Introdueix el nom del jugador: ').strip()

    # Filtra les dades de la prova (X_test) per identificar el jugador pel seu nom
    player_test_indices = X_test.filter(like=f'player_name_{player_name_input}', axis=1)

    # Comprova si hi ha coincidències amb el nom del jugador al conjunt de prova
    if not player_test_indices.empty:
        # Obtenim els índexs on apareix el jugador
        player_indices = player_test_indices.sum(axis=1).astype(bool)
        player_predictions = y_pred[player_indices]

        if len(player_predictions) > 0:
            print(f'Les puntuacions predites per al jugador {player_name_input} són: {player_predictions}')
        else:
            print(f'No s\'han trobat dades per al jugador {player_name_input} al conjunt de prova.')
    else:
        print(f'El jugador {player_name_input} no es troba al conjunt de prova.')
else:
    print(y_pred)

# 7. Avaluar el model
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

