# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:58:39 2024

@author: nildi
"""


# Importació de llibreries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import xgboost as xgb

# Carregar les dades des de l'arxiu CSV
df = pd.read_csv('./data/fantasy_data.csv')

# Comprovem les primeres files per assegurar-nos que les dades s'han carregat correctament
print(df.head())

# Carregar el DataFrame (df)
df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])

# Crear una nova columna per 'jornada' (a partir de 'kickoff_time')
df['round'] = df['kickoff_time'].dt.strftime('%Y-%m-%d')

# Ordenar les dades pel jugador i la jornada
df = df.sort_values(by=['player_id', 'round'])
def split_train_test_by_rounds(df):
    train_data = []
    test_data = []

    for player_id, player_data in df.groupby('player_id'):
        train_data.append(player_data.iloc[:18])  # Primeres 18 jornades
        test_data.append(player_data.iloc[18:])  # Darreres 18 jornades

    train_df = pd.concat(train_data)
    test_df = pd.concat(test_data)

    return train_df, test_df

# Dividim el dataset
train_df, test_df = split_train_test_by_rounds(df)

def create_features(df, n_prev_games):
    """
    Genera les característiques dels últims n_prev_games partits de manera vectoritzada.
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
    
    # Processar cada jugador de manera independent
    for player_id, player_data in df.groupby('player_id'):
        player_data = player_data.sort_values('round')
        
        for i in range(n_prev_games, len(player_data)):
            last_games = player_data.iloc[i-n_prev_games:i]
            
            X = {}
            for game_idx, game in enumerate(last_games.itertuples(), start=1):
                for col in feature_columns:
                    X[f'{col}_game_{game_idx}'] = getattr(game, col)
            
            current_game = player_data.iloc[i]
            X['player_id'] = player_id
            X['current_team'] = current_game['team']
            X['current_opponent'] = current_game['opponent_team']
            X['current_is_home'] = current_game['was_home']
            
            target_value = current_game['total_points']
            features_list.append(X)
            target_list.append(target_value)
    
    features_df = pd.DataFrame(features_list)
    return features_df, target_list

# [Les importacions i el codi fins a la funció evaluate_xgboost_parameters es mantenen igual...]

def evaluate_xgboost_parameters(df, n_prev_games_list, n_estimators_list, max_depth_list, learning_rate_list):
    """
    Avalua el model XGBoost amb diferents combinacions de paràmetres.
    """
    results = []
    
    # Calcular el total d'iteracions
    total_iterations = len(n_prev_games_list) * len(n_estimators_list) * len(max_depth_list) * len(learning_rate_list)
    
    # Barra de progrés principal
    main_progress = tqdm(total=total_iterations, desc="Progrés total XGBoost", position=0)
    
    for n_prev in tqdm(n_prev_games_list, desc="Processant n_prev_games", position=1, leave=False):
        print(f"\nCreant features per n_prev_games={n_prev}...")
        try:
            X, y = create_features(df, n_prev)
            
            categorical_columns = (
                [col for col in X.columns if 'team_game_' in col] + 
                [col for col in X.columns if 'opponent_team_game_' in col] +
                ['current_team', 'current_opponent']
            )
            X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
            
            X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
            
            print("Escalant dades...")
            scaler = StandardScaler()
            numeric_cols = [col for col in X_train.columns if col not in ['player_id']]
            X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
            X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
            X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
            
            for n_est in tqdm(n_estimators_list, desc="n_estimators", position=2, leave=False):
                for max_d in tqdm(max_depth_list, desc="max_depth", position=3, leave=False):
                    for lr in learning_rate_list:
                        try:
                            print(f"\nEntrenant XGBoost amb paràmetres:")
                            print(f"n_prev_games={n_prev}, n_estimators={n_est}, max_depth={max_d}, learning_rate={lr}")
                            
                            model = xgb.XGBRegressor(
                                n_estimators=n_est,
                                max_depth=max_d,
                                learning_rate=lr,
                                random_state=42,
                                tree_method='hist',
                                n_jobs=-1,
                                early_stopping_rounds=50  # Moure aquí
                            )
                            
                            # Modificar el fit per utilitzar eval_set correctament
                            model.fit(
                                X_train, 
                                y_train,
                                eval_set=[(X_val, y_val)],
                                verbose=False
                            )
                            
                            # Prediccions
                            y_val_pred = model.predict(X_val)
                            y_test_pred = model.predict(X_test)
                            
                            # Càlcul de mètriques
                            val_mae = mean_absolute_error(y_val, y_val_pred)
                            val_mse = mean_squared_error(y_val, y_val_pred)
                            val_rmse = np.sqrt(val_mse)
                            test_mae = mean_absolute_error(y_test, y_test_pred)
                            test_mse = mean_squared_error(y_test, y_test_pred)
                            test_rmse = np.sqrt(test_mse)
                            
                            print(f"Resultats parcials - VAL MAE: {val_mae:.4f}, TEST MAE: {test_mae:.4f}")
                            
                            result_dict = {
                                'n_prev_games': n_prev,
                                'n_estimators': n_est,
                                'max_depth': max_d,
                                'learning_rate': lr,
                                'validation_MAE': val_mae,
                                'validation_MSE': val_mse,
                                'validation_RMSE': val_rmse,
                                'test_MAE': test_mae,
                                'test_MSE': test_mse,
                                'test_RMSE': test_rmse,
                                'best_iteration': model.best_iteration
                            }
                            
                            results.append(result_dict)
                            
                        except Exception as e:
                            print(f"Error amb paràmetres: n_prev={n_prev}, n_est={n_est}, max_d={max_d}, lr={lr}")
                            print(f"Error: {str(e)}")
                            continue
                        
                        main_progress.update(1)
        
        except Exception as e:
            print(f"Error processant n_prev_games={n_prev}")
            print(f"Error: {str(e)}")
            continue
    
    main_progress.close()
    
    # Crear DataFrame només si tenim resultats
    if results:
        results_df = pd.DataFrame(results)
        # Ordenar per validation_MAE en lloc de val_MAE
        results_df = results_df.sort_values('validation_MAE')
        return results_df
    else:
        print("No s'han obtingut resultats")
        return pd.DataFrame()


def plot_error_evolution(results_df):
    """
    Crea una gràfica que mostra l'evolució dels errors segons n_prev_games
    """
    plt.figure(figsize=(12, 6))
    
    # Calcular la mitjana dels errors per cada n_prev_games
    error_means = results_df.groupby('n_prev_games').agg({
        'validation_MAE': 'mean',
        'validation_RMSE': 'mean',
        'test_MAE': 'mean',
        'test_RMSE': 'mean'
    }).reset_index()
    
    # Crear les línies per cada tipus d'error
    plt.plot(error_means['n_prev_games'], error_means['validation_MAE'], 
             marker='o', label='Validation MAE', linestyle='-', color='blue')
    plt.plot(error_means['n_prev_games'], error_means['validation_RMSE'], 
             marker='s', label='Validation RMSE', linestyle='-', color='red')
    plt.plot(error_means['n_prev_games'], error_means['test_MAE'], 
             marker='o', label='Test MAE', linestyle='--', color='lightblue')
    plt.plot(error_means['n_prev_games'], error_means['test_RMSE'], 
             marker='s', label='Test RMSE', linestyle='--', color='lightcoral')

    # Configurar la gràfica
    plt.xlabel('Nombre de partits previs')
    plt.ylabel('Error')
    plt.title('Evolució dels errors segons el nombre de partits previs')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Afegir els valors a cada punt
    for metric in ['validation_MAE', 'validation_RMSE', 'test_MAE', 'test_RMSE']:
        for i, row in error_means.iterrows():
            plt.annotate(f'{row[metric]:.3f}', 
                        (row['n_prev_games'], row[metric]),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center',
                        fontsize=8)
    
    plt.tight_layout()
    return plt



# Paràmetres per XGBoost
n_prev_games_list = [3,4,5,6,7,8,9,10]  # Després provar amb 5 i 10
n_estimators_list = [500]
max_depth_list = [6]  # XGBoost normalment funciona millor amb arbres més petits
learning_rate_list = [0.05]

# Executar avaluació per XGBoost
xgb_results = evaluate_xgboost_parameters(
    df,
    n_prev_games_list,
    n_estimators_list,
    max_depth_list,
    learning_rate_list
)

# Guardar resultats només si tenim resultats
# Després d'executar l'avaluació de XGBoost, afegeix:
if not xgb_results.empty:
    xgb_results.to_csv('XGBoost_evaluation_results.csv', index=False)
    print("\nMillors resultats XGBoost:")
    print(xgb_results.head())
    
    # Crear i guardar la gràfica
    plt = plot_error_evolution(xgb_results)
    plt.savefig('error_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
else:
    print("No s'han pogut generar resultats")