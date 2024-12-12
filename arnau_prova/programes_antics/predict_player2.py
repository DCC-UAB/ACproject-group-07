import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import os

# Carregar les dades des de l'arxiu CSV
df = pd.read_csv('./data/fantasy_data.csv')

# Preparar les dades
df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])
df['round'] = df['kickoff_time'].dt.strftime('%Y-%m-%d')

def assign_round_numbers(df):
    """Assigna números de jornada (1-18) per cada meitat de temporada"""
    df = df.sort_values(by=['player_id', 'kickoff_time'])
    
    # Crear números de jornada originals (1-36)
    df['original_round'] = df.groupby('player_id').cumcount() + 1
    
    # Crear números de jornada normalitzats (1-18)
    df['round_number'] = df['original_round'].apply(lambda x: x if x <= 18 else x - 18)
    
    return df

df = assign_round_numbers(df)

def create_features(df, target_round):
    """
    Genera característiques utilitzant tots els partits anteriors fins a target_round
    """
    features_list = []
    target_list = []
    
    feature_columns = [
        'assists', 'bonus', 'bps', 'clean_sheets', 'creativity',
        'goals_conceded', 'goals_scored', 'ict_index', 'influence',
        'minutes', 'own_goals','red_cards', 'saves', 'selected', 'team_a_score', 'team_h_score',
        'threat', 'transfers_balance', 'transfers_in', 'transfers_out',
        'value', 'was_home'
    ]
    
    n_prev_games = target_round - 1
    
    # Processar cada jugador
    for player_id, player_data in df.groupby('player_id'):
        player_data = player_data.sort_values('round_number')
        
        # Només processar si tenim la jornada objectiu
        target_data = player_data[player_data['round_number'] == target_round]
        if not target_data.empty:
            # Agafar els partits anteriors
            prev_games = player_data[player_data['round_number'] < target_round]
            if len(prev_games) == n_prev_games:  # Només si tenim totes les jornades anteriors
                X = {}
                for game_idx, game in enumerate(prev_games.itertuples(), start=1):
                    for col in feature_columns:
                        X[f'{col}_game_{game_idx}'] = getattr(game, col)
                
                # Afegir informació del partit actual
                current_game = target_data.iloc[0]
                X['player_id'] = player_id
                X['current_team'] = current_game['team']
                X['current_opponent'] = current_game['opponent_team']
                X['current_is_home'] = current_game['was_home']
                
                features_list.append(X)
                target_list.append(current_game['total_points'])
    
    features_df = pd.DataFrame(features_list)
    return features_df, target_list

def process_features(features_df, scaler=None, is_training=True):
    """
    Processa les característiques: crea dummies i escala les variables numèriques
    """
    # Identificar columnes categòriques
    categorical_columns = (
        [col for col in features_df.columns if 'team_game_' in col] + 
        [col for col in features_df.columns if 'opponent_team_game_' in col] +
        ['current_team', 'current_opponent']
    )
    
    # Crear dummies per les variables categòriques
    features_df = pd.get_dummies(features_df, columns=categorical_columns, drop_first=True)
    
    # Escalar variables numèriques
    numeric_cols = [col for col in features_df.columns if col not in ['player_id']]
    
    if is_training:
        scaler = StandardScaler()
        features_df[numeric_cols] = scaler.fit_transform(features_df[numeric_cols])
        return features_df, scaler
    else:
        features_df[numeric_cols] = scaler.transform(features_df[numeric_cols])
        return features_df

# Demanar la jornada a predir
print("\nJornades disponibles: 1-18")
target_round = int(input("Introdueix el número de la jornada (1-18) que vols predir: "))

if target_round < 2 or target_round > 18:
    print("Jornada no vàlida. Ha de ser entre 2 i 18.")
    exit()

# Dividir en train i test
train_df = df[df['original_round'] <= 18].copy()
test_df = df[df['original_round'] > 18].copy()

# Comprovar si el model i l'scaler existeixen
model_path = f'fantasy_model_round_{target_round}.joblib'
scaler_path = f'fantasy_scaler_round_{target_round}.joblib'

if os.path.exists(model_path) and os.path.exists(scaler_path):
    print("Carregant model i scaler existents...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
else:
    print("Entrenant nou model...")
    # Crear i processar features per training
    X_train, y_train = create_features(train_df, target_round)
    
    if len(X_train) == 0:
        print(f"No hi ha prou dades per entrenar el model per la jornada {target_round}")
        exit()
    
    # Processar features
    X_train_processed, scaler = process_features(X_train, is_training=True)
    
    # Entrenar model
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    
    model.fit(X_train_processed, y_train)
    
    # Guardar model i scaler
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print("Model i scaler guardats correctament")

# Funcionalitat per predir punts
# Funcionalitat per predir punts
while True:
    try:
        player_id_input = input("\nIntrodueix l'ID del jugador (o 'q' per sortir): ")
        if player_id_input.lower() == 'q':
            break

        # Comprovar si l'ID del jugador existeix
        player_id = int(player_id_input)
        player_data = test_df[test_df['player_id'] == player_id]

        if player_data.empty:
            print("No s'ha trobat dades per a l'ID de jugador especificat.")
            continue

        # Obtenir el nom del jugador
        player_name = player_data['player_name'].iloc[0]

        # Crear i processar features per la predicció
        X_player, _ = create_features(player_data, target_round)

        if X_player.empty:
            print(f"No hi ha prou dades històriques per predir la jornada {target_round}")
            continue

        # Processar features
        X_player_processed = process_features(X_player, scaler=scaler, is_training=False)

        # Fer predicció
        predicted_points = model.predict(X_player_processed)
        real_points = player_data[player_data['round_number'] == target_round]['total_points'].values[0]
        real_round = player_data[player_data['round_number'] == target_round]['original_round'].values[0]

        # Mostra la predicció amb el nom i ID del jugador
        print(f"\nPredicció per al jugador {player_name} (ID: {player_id}):")
        print(f"Jornada predita: {target_round} (Jornada real: {real_round})")
        print(f"Punts predits: {predicted_points[0]:.2f}")
        print(f"Punts reals: {real_points}")

    except ValueError:
        print("Introdueix un número vàlid per a l'ID del jugador.")
    except Exception as e:
        print(f"Error: {e}")
