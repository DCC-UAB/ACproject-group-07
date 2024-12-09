# Importació de llibreries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

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
    """
    Genera les característiques dels últims n_prev_games partits de manera vectoritzada.
    """
    # Inicialitzar llistes per a features i target
    features_list = []
    target_list = []
    
    # Llista de columnes que es faran servir com a features
    feature_columns = [
        'assists', 'bonus', 'bps', 'clean_sheets', 'creativity',
        'goals_conceded', 'goals_scored', 'ict_index', 'influence',
        'minutes', 'own_goals','red_cards', 'saves', 'selected', 'team_a_score', 'team_h_score',
        'threat', 'transfers_balance', 'transfers_in', 'transfers_out',
        'value', 'was_home'
    ]
    
    # Processar cada jugador de manera independent
    for player_id, player_data in df.groupby('player_id'):
        # Ordenar les dades per 'round' (jornada)
        player_data = player_data.sort_values('round')
        
        # Crear arrays per emmagatzemar les característiques dels partits anteriors
        for i in range(n_prev_games, len(player_data)):
            # Seleccionem els darrers n_prev_games partits
            last_games = player_data.iloc[i-n_prev_games:i]
            
            # Generar un diccionari de features
            X = {}
            for game_idx, game in enumerate(last_games.itertuples(), start=1):
                for col in feature_columns:
                    X[f'{col}_game_{game_idx}'] = getattr(game, col)
            
            # Afegir informació del partit actual
            current_game = player_data.iloc[i]
            X['player_id'] = player_id
            X['current_team'] = current_game['team']
            X['current_opponent'] = current_game['opponent_team']
            X['current_is_home'] = current_game['was_home']
            
            # Afegir target (total points del partit actual)
            target_value = current_game['total_points']
            
            features_list.append(X)
            target_list.append(target_value)
    
    # Convertir les característiques a DataFrame
    features_df = pd.DataFrame(features_list)
    return features_df, target_list

def evaluate_model_parameters(df, n_prev_games_list, n_estimators_list, max_depth_list, min_samples_split_list):
    """
    Avalua el model amb diferents combinacions de paràmetres, utilitzant validació.
    """
    results = []
    
    # Calcular el total d'iteracions
    total_iterations = len(n_prev_games_list) * len(n_estimators_list) * len(max_depth_list) * len(min_samples_split_list)
    pbar = tqdm(total=total_iterations, desc="Avaluant models")
    
    for n_prev in n_prev_games_list:
        print(f"\nProcessant n_prev_games={n_prev}...")
        # Crear features
        X, y = create_features(df, n_prev)
        
        # Convert categorical variables to dummies
        categorical_columns = (
            [col for col in X.columns if 'team_game_' in col] + 
            [col for col in X.columns if 'opponent_team_game_' in col] +
            ['current_team', 'current_opponent']
        )
        X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
        
        # Primer split: separar el test set
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Segon split: separar training i validation
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
        
        # Escalar les dades numèriques
        scaler = StandardScaler()
        numeric_cols = [col for col in X_train.columns if col not in ['player_id']]
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
        
        for n_est in n_estimators_list:
            for max_d in max_depth_list:
                for min_samples in min_samples_split_list:
                    # Entrenar model
                    rf = RandomForestRegressor(
                        n_estimators=n_est,
                        max_depth=max_d,
                        min_samples_split=min_samples,
                        random_state=42
                    )
                    try:
                        print(f"Dimensions de X_train: {X_train.shape}")  
                        print(f"Dimensions de y_train: {len(y_train)}")
                        print(f"Dimensions de y_val: {len(y_val)}")                    
                        # Entrenem amb dades d'entrenament
                        rf.fit(X_train, y_train)
                        
                        # Prediccions en validació
                        y_val_pred = rf.predict(X_val)
                        val_mae = mean_absolute_error(y_val, y_val_pred)
                        val_mse = mean_squared_error(y_val, y_val_pred)
                        val_rmse = np.sqrt(val_mse)
                        
                        # Prediccions en test
                        y_test_pred = rf.predict(X_test)
                        test_mae = mean_absolute_error(y_test, y_test_pred)
                        test_mse = mean_squared_error(y_test, y_test_pred)
                        test_rmse = np.sqrt(test_mse)
                        
                    except Exception as e:
                        print(f"Error amb paràmetres: n_prev={n_prev}, n_est={n_est}, max_d={max_d}, min_samples={min_samples}")
                        print(f"Error: {str(e)}")
                        continue
                    
                    results.append({
                        'n_prev_games': n_prev,
                        'n_estimators': n_est,
                        'max_depth': str(max_d),
                        'min_samples_split': min_samples,
                        'val_MAE': val_mae,
                        'val_MSE': val_mse,
                        'val_RMSE': val_rmse,
                        'test_MAE': test_mae,
                        'test_MSE': test_mse,
                        'test_RMSE': test_rmse
                    })

                    pbar.update(1)
    
    pbar.close()
    # Convertir a DataFrame i ordenar per validation MAE
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('val_MAE')
    
    return results_df


n_prev_games_list = [2] #despres fer per 5 i 10
n_estimators_list = [1000]
max_depth_list = [10]
min_samples_split_list = [2]

# Executar l'avaluació
results = evaluate_model_parameters(df,n_prev_games_list, n_estimators_list, max_depth_list, min_samples_split_list)

# Opcional: Guardar els resultats en un CSV
results.to_csv('RFG_evaluation_results.csv', index=False)