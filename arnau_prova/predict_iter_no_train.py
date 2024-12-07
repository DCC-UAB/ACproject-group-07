import pandas as pd
import numpy as np
import joblib

# Carregar les dades des de l'arxiu CSV
df = pd.read_csv('./data/fantasy_data.csv')

# Convertir kickoff_time a datetime i crear round
df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])
df['round'] = df['kickoff_time'].dt.strftime('%Y-%m-%d')

# Ordenar les dades pel jugador i la jornada
df = df.sort_values(by=['player_id', 'round'])

def predict_player_points(df, player_name=None, player_id=None, n_prev_games=5):
    """
    Prediu els punts esperats per un jugador específic utilitzant el model entrenat.
    """
    # Carregar el model i les columnes
    model = joblib.load('saved_model/model_rf.joblib')
    model_columns = joblib.load('saved_model/model_columns.joblib')
    
    # Filtrar per jugador
    if player_name is not None:
        player_data = df[df['player_name'] == player_name].copy()
    elif player_id is not None:
        player_data = df[df['player_id'] == player_id].copy()
    else:
        raise ValueError("Has d'especificar player_name o player_id")
    
    if len(player_data) == 0:
        raise ValueError("No s'han trobat dades per aquest jugador")
        
    # Ordenar per data
    player_data = player_data.sort_values('round')
    
    # Obtenir les últimes n jornades
    if len(player_data) < n_prev_games:
        raise ValueError(f"No hi ha prou dades. Es necessiten {n_prev_games} partits i només n'hi ha {len(player_data)}")
    
    # Crear features pel jugador
    numeric_cols = ['assists', 'bonus', 'bps', 'clean_sheets', 'creativity',
                   'goals_conceded', 'goals_scored', 'ict_index', 'influence', 
                   'minutes', 'own_goals', 'penalties_missed', 'penalties_saved', 
                   'red_cards', 'saves', 'selected', 'team_a_score', 'team_h_score', 
                   'threat', 'transfers_balance', 'transfers_in', 'transfers_out', 
                   'value', 'yellow_cards', 'ppm']
    
    # Agafar les últimes n jornades
    last_games = player_data.iloc[-n_prev_games:]
    current_game = player_data.iloc[-1]  # Utilitzem l'últim partit com a referència
    
    # Crear features
    X_numeric = last_games[numeric_cols].values.flatten()
    additional_features = [
        current_game['player_id'],
        current_game['player_name'],
        current_game['team'],
        current_game['opponent_team'],
        1 if current_game['was_home'] else 0
    ]
    
    # Combinar features
    X = np.concatenate([X_numeric, additional_features])
    
    # Crear DataFrame amb els noms de columnes correctes
    feature_names = []
    for i in range(n_prev_games):
        for col in numeric_cols:
            feature_names.append(f"{col}_game_{i+1}")
    feature_names.extend(['player_id', 'player_name', 'team', 'opponent_team', 'was_home'])
    
    X_df = pd.DataFrame([X], columns=feature_names)
    
    # Convertir a dummies
    X_df = pd.get_dummies(X_df, columns=['player_name', 'team', 'opponent_team'])
    
    # Assegurar que tenim totes les columnes necessàries
    missing_cols = {col: 0 for col in model_columns if col not in X_df.columns}
    X_df = pd.concat([X_df, pd.DataFrame(missing_cols, index=X_df.index)], axis=1)
    
    # Reordenar columnes per coincidir amb el model
    X_df = X_df[model_columns]
    
    # Fer la predicció
    prediction = model.predict(X_df)[0]
    
    return prediction

# Per nom
punts = predict_player_points(df, player_name="Aaron Connolly")
print(f"Punts esperats per Aaron Connolly: {punts:.2f}")
