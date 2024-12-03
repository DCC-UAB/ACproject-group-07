# Importació de llibreries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

def load_and_preprocess_data(file_path):
    """Carrega i preprocessa el dataset."""
    df = pd.read_csv(file_path)
    df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])
    df['round'] = df['kickoff_time'].dt.strftime('%Y-%m-%d')  # Identificar les jornades
    df = df.sort_values(by=['player_id', 'round'])
    return df

def create_features(df, n_prev_games):
    """Crea les característiques i la variable objectiu utilitzant les darreres n jornades."""
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
            
            # Afegir identificadors categòrics
            X['player_id'] = player_id
            X['player_name'] = player_data.iloc[i]['player_name']
            X['team'] = player_data.iloc[i]['team']
            
            target_value = player_data.iloc[i]['total_points']
            features.append(X)
            target.append(target_value)
    
    return pd.DataFrame(features), target

def train_model(X, y):
    """Entrena un model Random Forest."""
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    return rf

def iterative_predictions_for_all(df, model, n_prev_games, n_max_games):
    """
    Realitza prediccions iteratives per a tots els jugadors del dataset.
    """
    all_predictions = {}

    for player_id in df['player_id'].unique():
        player_data = df[df['player_id'] == player_id].sort_values(by='round')

        # Comprovar si el jugador té prou partits
        if len(player_data) < n_max_games:
            continue  # Ometre jugadors amb menys partits disponibles

        predictions = []
        current_window = player_data.iloc[:n_prev_games].copy()

        for i in range(n_prev_games, n_max_games):
            # Ensure the current window has the same features as the training set
            X_current = current_window[['assists', 'bonus', 'bps', 'clean_sheets', 'creativity', 'goals_conceded',
                                        'goals_scored', 'ict_index', 'influence', 'minutes', 
                                        'own_goals', 'penalties_missed', 'penalties_saved', 'red_cards', 'saves', 
                                        'selected', 'team_a_score', 'team_h_score', 'threat', 'transfers_balance', 
                                        'transfers_in', 'transfers_out', 'value', 'was_home', 'yellow_cards']].mean().values.reshape(1, -1)

            # Add missing features to match the training set
            for col in model.feature_importances_:
                if col not in X_current.columns:
                    X_current[col] = 0  # or any default value

            y_pred = model.predict(X_current)[0]
            predictions.append(y_pred)

            new_row = player_data.iloc[i].copy()
            new_row['total_points'] = y_pred
            current_window = pd.concat([current_window.iloc[1:], pd.DataFrame([new_row])])

        all_predictions[player_id] = predictions

    return all_predictions

def main():
    # Paràmetres
    file_path = './data/fantasy_data.csv'
    n_prev_games = 5
    n_max_games = 8

    # 1. Carregar i preprocessar les dades
    df = load_and_preprocess_data(file_path)

    # 2. Crear característiques i variable objectiu
    X, y = create_features(df, n_prev_games)
    X = pd.get_dummies(X, columns=['player_name', 'team'], drop_first=True)

    # 3. Dividir les dades en entrenament i prova
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Entrenar el model
    rf = train_model(X_train, y_train)

    # 5. Predicció en el conjunt de prova
    y_pred = rf.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Error mitjà absolut en el conjunt de prova: {mae}")

    # 6. Prediccions iteratives per a tots els jugadors
    all_predictions = iterative_predictions_for_all(df, rf, n_prev_games, n_max_games)
    print("Prediccions iteratives per a tots els jugadors:")
    for player_id, predictions in all_predictions.items():
        print(f"Jugador {player_id}: {predictions}")

if __name__ == "__main__":
    main()
