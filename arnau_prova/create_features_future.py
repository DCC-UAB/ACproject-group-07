
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import xgboost as xgb

def create_features(df, n_prev_games, n_future=1):
    """
    Genera features per predir partits en el futur.
    
    Paràmetres:
    - df: DataFrame amb les dades dels partits
    - n_prev_games: Número de partits anteriors a utilitzar com a features
    - n_future: Número de partits endavant a predir (1 = següent partit, 2 = d'aquí dos partits, etc.)
    
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
    
    features_list = []
    player_info_list = []
    
    # Ordenar dades per jugador i data
    df = df.sort_values(by=['player_id', 'kickoff_time']).reset_index(drop=True)
    
    # Iterar per cada jugador
    for player_id, player_data in df.groupby('player_id'):
        player_data = player_data.reset_index(drop=True)
        
        # Iterar per cada possible finestra de predicció
        # Ara necessitem n_prev_games + n_future partits disponibles
        for i in range(n_prev_games, len(player_data) - n_future + 1):
            # Agafar els n_prev_games anteriors
            previous_games = player_data.iloc[i-n_prev_games:i]
            # Agafar el partit que volem predir (n_future partits endavant)
            target_game = player_data.iloc[i + n_future - 1]
            
            # Crear diccionari per aquesta predicció
            X = {
                'player_id': player_id,
                'target_total_points': target_game['total_points'],
                'prediction_game_number': i + n_future - 1,
                'prediction_date': target_game['kickoff_time']
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
                    'player_name': target_game['player_name'],
                    'team': target_game['team']
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

# Exemple d'ús:
if __name__ == "__main__":
    # Carregar dades
    df = pd.read_csv('./data/fantasy_data.csv')
    
    n_prev_games =2
    n_future=3
    # Crear features per predir el partit d'aquí a 3 jornades
    features_df, player_info_df = create_features(df, n_prev_games, n_future)
    
    # Guardar les features
    features_df.to_csv(f'data/features_df_future_{n_future}.csv', index=False)

    print(features_df.shape)
    print(features_df.head())