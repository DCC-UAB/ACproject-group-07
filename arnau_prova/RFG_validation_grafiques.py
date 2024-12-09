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

from sklearn.model_selection import TimeSeriesSplit, cross_val_score

def evaluate_model_parameters(df, n_prev_games_list, n_estimators_list, max_depth_list, min_samples_split_list, cv_folds=5):
    """
    Avalua el model amb diferents combinacions de paràmetres, utilitzant TimeSeriesSplit cross-validation.
    """
    results = []
    best_model = None
    best_X_test, best_y_test, best_y_pred = None, None, None  # Per guardar les prediccions del millor model
    pbar = tqdm(total=len(n_prev_games_list) * len(n_estimators_list) * len(max_depth_list) * len(min_samples_split_list),
                desc="Avaluant models")
    
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
        
        # Primer split per conjunt de test
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Segon split per entrenament i validació
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
        
        # TimeSeriesSplit per cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        for n_est in n_estimators_list:
            for max_d in max_depth_list:
                for min_samples in min_samples_split_list:
                    rf = RandomForestRegressor(
                        n_estimators=n_est,
                        max_depth=max_d,
                        min_samples_split=min_samples,
                        random_state=42
                    )
                    try:
                        # Entrenar el model
                        rf.fit(X_train, y_train)
                        
                        # Cross-validation
                        cv_scores = cross_val_score(rf, X_train, y_train, cv=tscv, scoring='neg_mean_absolute_error')
                        mean_cv_score = -np.mean(cv_scores)
                        std_cv_score = np.std(cv_scores)
                        
                        # Prediccions per a validació
                        y_val_pred = rf.predict(X_val)
                        val_mae = mean_absolute_error(y_val, y_val_pred)
                        
                        # Si és el millor model, guarda'l
                        if best_model is None or val_mae < min([r.get('val_MAE', float('inf')) for r in results]):
                            best_model = rf
                            best_X_test, best_y_test = X_test, y_test
                            best_y_pred = rf.predict(X_test)
                    
                    except Exception as e:
                        print(f"Error amb cross-validation: n_prev={n_prev}, n_est={n_est}, max_d={max_d}, min_samples={min_samples}")
                        print(f"Error: {str(e)}")
                        continue
                    
                    results.append({
                        'n_prev_games': n_prev,
                        'n_estimators': n_est,
                        'max_depth': str(max_d),
                        'min_samples_split': min_samples,
                        'mean_cv_MAE': mean_cv_score,
                        'std_cv_MAE': std_cv_score,
                        'val_MAE': val_mae
                    })

                    pbar.update(1)
    
    pbar.close()
    results_df = pd.DataFrame(results)
    return results_df, best_y_test, best_y_pred

def plot_temporal_results(y_true, y_pred, title="Resultats Temporals"):
    """
    Gràfic de línia per comparar valors reals i prediccions al llarg de les jornades.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label="Valors Reals", marker='o', linestyle='-', alpha=0.7)
    plt.plot(y_pred, label="Prediccions", marker='x', linestyle='--', alpha=0.7)
    plt.xlabel("Índex de Mostra")
    plt.ylabel("Punts")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_predictions_vs_actuals(y_true, y_pred, title="Prediccions vs Valors Reals"):
    """
    Gràfic de dispersió per comparar les prediccions amb els valors reals.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, color='blue')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label="Línia ideal (y=x)")
    plt.xlabel("Valors Reals")
    plt.ylabel("Prediccions")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()



n_prev_games_list = [2] #despres fer per 5 i 10
n_estimators_list = [1000]
max_depth_list = [10]
min_samples_split_list = [2]

# Executar l'avaluació
results, y_test, y_pred = evaluate_model_parameters(df, n_prev_games_list, n_estimators_list, max_depth_list, min_samples_split_list)

# Gràfic Temporal
plot_temporal_results(y_test, y_pred, title="Resultats Temporals (Prediccions vs Reals)")

# Scatter Plot
plot_predictions_vs_actuals(y_test, y_pred, title="Prediccions vs Valors Reals")
