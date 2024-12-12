# Importació de llibreries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import xgboost as xgb

#get noms players del test
def get_test_player_names(features_df, player_info_df, test_indices):
    """
    Retorna una llista amb els noms de jugadors que es troben al conjunt de test.

    Paràmetres:
    - features_df: DataFrame original amb les característiques generades.
    - player_info_df: DataFrame amb la informació dels jugadors (`player_id`, `player_name`, `team`).
    - test_indices: Índexs del conjunt de test.

    Retorna:
    - Llista de noms únics de jugadors al conjunt de test.
    """
    # Obtenir els player_id presents al conjunt de test
    test_player_ids = features_df.loc[test_indices, 'player_id'].unique()
    
    # Filtrar la informació dels jugadors basant-se en els player_id
    test_players = player_info_df[player_info_df['player_id'].isin(test_player_ids)]
    
    # Retornar els noms dels jugadors
    return test_players['player_name'].unique()

#get la info dels players
def get_player_info(player_info_df, player_name):
    """
    Retorna informació del jugador basant-se en el seu nom.

    Paràmetres:
    - player_info_df: DataFrame amb la informació dels jugadors (`player_id`, `player_name`, `team`).
    - player_name: Nom del jugador a buscar.

    Retorna:
    - Diccionari amb la informació del jugador (`player_id`, `player_name`, `team`).
    """
    player_row = player_info_df[player_info_df['player_name'] == player_name]
    
    if not player_row.empty:
        return player_row.iloc[0].to_dict()
    else:
        print(f"El jugador amb nom '{player_name}' no s'ha trobat.")
        return None

# Per carregar els DataFrames en futures execucions:
def load_or_create_features(df, n_prev_games, force_create=False):
    """
    Carrega les features des dels arxius CSV o les crea si no existeixen.
    
    Paràmetres:
    - df: DataFrame original
    - n_prev_games: Número de partits anteriors a utilitzar
    - force_create: Si és True, força la creació de noves features encara que existeixin els arxius
    
    Retorna:
    - features_df: DataFrame amb les features
    - player_info_df: DataFrame amb la informació dels jugadors
    """
    try:
        if force_create:
            raise FileNotFoundError  # Forçar la creació de noves features
            
        print("Carregant features des dels arxius...")
        features_df = pd.read_csv('data/features_df.csv')
        player_info_df = pd.read_csv('data/player_info_df.csv')
        
        # Convertir la columna de dates a datetime
        features_df['prediction_date'] = pd.to_datetime(features_df['prediction_date'])
        
        print("Features carregades correctament!")
        
    except FileNotFoundError:
        print("Creant noves features...")
        features_df, player_info_df = create_features(df, n_prev_games)
        
        # Guardar els DataFrames
        features_df.to_csv('data/features_df.csv', index=False)
        player_info_df.to_csv('data/player_info_df.csv', index=False)
        
        print("Features creades i guardades!")
    
    return features_df, player_info_df

# Carregar les dades des de l'arxiu CSV
def create_features(df, n_prev_games):
    """
    Genera múltiples files per jugador, una per cada possible predicció utilitzant els n_prev_games anteriors.
    
    Paràmetres:
    - df: DataFrame amb les dades dels partits
    - n_prev_games: Número de partits anteriors a utilitzar com a features
    
    Retorna:
    - features_df: DataFrame amb les features generades
    - player_info_df: DataFrame amb la informació dels jugadors
    """
    feature_columns = [
        'assists', 'bonus', 'bps', 'clean_sheets', 'creativity',
        'goals_conceded', 'goals_scored', 'ict_index', 'influence',
        'minutes', 'own_goals', 'red_cards', 'saves', 'selected',
        'team_a_score', 'team_h_score', 'threat', 'transfers_balance',
        'transfers_in', 'transfers_out', 'value', 'was_home', 'opponent_team', 'total_points'
    ]
    
    # Llistes per emmagatzemar resultats
    features_list = []
    player_info_list = []
    
    # Ordenar dades per jugador i data
    df = df.sort_values(by=['player_id', 'kickoff_time']).reset_index(drop=True)
    
    # Iterar per cada jugador
    for player_id, player_data in df.groupby('player_id'):
        player_data = player_data.reset_index(drop=True)
        
        # Iterar per cada possible finestra de predicció
        for i in range(n_prev_games, len(player_data)):
            # Agafar els n_prev_games anteriors i el partit actual
            previous_games = player_data.iloc[i-n_prev_games:i]
            current_game = player_data.iloc[i]
            
            # Crear diccionari per aquesta predicció
            X = {
                'player_id': player_id,
                'target_total_points': current_game['total_points'],
                'prediction_game_number': i,
                'prediction_date': current_game['kickoff_time']
            }
            
            # Afegir features dels partits anteriors
            for game_idx, game in enumerate(previous_games.itertuples(), start=1):
                for col in feature_columns:
                    X[f'{col}_game_{game_idx}'] = getattr(game, col)
            
            features_list.append(X)
            
            # Guardar informació del jugador (només un cop)
            if len(player_info_list) == 0 or not any(p['player_id'] == player_id for p in player_info_list):
                player_info = {
                    'player_id': player_id,
                    'player_name': current_game['player_name'],
                    'team': current_game['team']
                }
                player_info_list.append(player_info)
    
    # Convertir a DataFrames
    features_df = pd.DataFrame(features_list)
    opponent_cols = [col for col in features_df.columns if 'opponent_team' in col]
    features_df = pd.get_dummies(features_df, columns=opponent_cols, drop_first=True)
    
    player_info_df = pd.DataFrame(player_info_list)
    
    print(f"Total de files generades: {len(features_df)}")
    print(f"Total de jugadors: {len(player_info_df)}")
    print(f"Mitjana de prediccions per jugador: {len(features_df)/len(player_info_df):.1f}")
    
    return features_df, player_info_df

