# train_model.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
from data_processing import load_or_create_features, get_test_player_names, get_player_info

def train_and_predict(features_df, val_size=0.2, test_size=0.2, random_state=42):
    """
    Prepara les dades, entrena el model i fa prediccions.
    
    Retorna:
    - model: Model entrenat
    - predictions: Prediccions de validació i test
    - data_splits: Conjunts de dades per avaluació
    """
    # Preparar dades
    X = features_df.drop(columns=['player_id', 'target_total_points', 'prediction_game_number', 'prediction_date'])
    y = features_df['target_total_points']

    # Separar en train, validation i test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size, random_state=random_state)

    print("\nDimensions dels conjunts:")
    print(f"Train set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")

    # Entrenar model
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=random_state,
        early_stopping_rounds=10,
        eval_metric='mae'
    )
    
    # Entrenar amb early stopping
    model.fit(
        X_train, 
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

    # Fer prediccions només per validació i test
    predictions = {
        'val': model.predict(X_val),
        'test': model.predict(X_test)
    }
    
    # Guardar dades necessàries per avaluació
    data_splits = {
        'y_val': y_val,
        'y_test': y_test,
        'X_val': X_val,
        'X_test': X_test
    }
    
    return model, predictions, data_splits

def evaluate_errors(predictions, data_splits):
    """
    Calcula i mostra els errors de validació i test.
    """
    metrics = {}
    
    # Calcular mètriques només per validació i test
    for split in ['val', 'test']:
        y_true = data_splits[f'y_{split}']
        y_pred = predictions[split]
        print(y_pred.shape)
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        metrics[split] = {
            'mae': mae,
            'rmse': rmse
        }
    
    # Mostrar resultats
    print("\nResultats de l'avaluació:")
    for split, metric in metrics.items():
        print(f"\n{split.capitalize()}:")
        print(f"MAE: {metric['mae']:.2f}")
        print(f"RMSE: {metric['rmse']:.2f}")
    
    return metrics


def create_enhanced_features(df, n_prev_games, n_future=1):
    """
    Crea features avançades per millorar la predicció d'actuacions excepcionals.
    """
    # Primer calculem estadístiques per jugador
    player_stats = {}
    for player_id, player_data in df.groupby('player_id'):
        player_stats[player_id] = {
            'mean_points': player_data['total_points'].mean(),
            'std_points': player_data['total_points'].std(),
            'max_points': player_data['total_points'].max(),
            'consistency': player_data['total_points'].std() / player_data['total_points'].mean(),
            'high_score_freq': (player_data['total_points'] > player_data['total_points'].mean() + 
                              player_data['total_points'].std()).mean()
        }
    
    feature_list = []
    player_info_list = []
    
    for player_id, player_data in df.groupby('player_id'):
        player_data = player_data.sort_values('kickoff_time').reset_index(drop=True)
        
        for i in range(n_prev_games, len(player_data) - n_future + 1):
            # Features bàsiques dels partits anteriors
            previous_games = player_data.iloc[i-n_prev_games:i]
            target_game = player_data.iloc[i + n_future - 1]
            
            X = {
                'player_id': player_id,
                'target_total_points': target_game['total_points'],
                'prediction_game_number': i + n_future - 1,
                'prediction_date': target_game['kickoff_time']
            }
            
            # Afegir features bàsiques dels partits anteriors
            for game_idx, game in enumerate(previous_games.itertuples(), start=1):
                for col in ['assists', 'bonus', 'bps', 'goals_scored', 'minutes', 'ict_index']:
                    X[f'{col}_game_{game_idx}'] = getattr(game, col)
            
            # Features de rendiment recent
            recent_points = previous_games['total_points'].values
            X.update({
                'recent_mean': np.mean(recent_points),
                'recent_std': np.std(recent_points),
                'recent_trend': np.polyfit(range(len(recent_points)), recent_points, 1)[0],
                'recent_max': np.max(recent_points),
                'games_since_exceptional': len(previous_games) - 
                    np.argmax(recent_points > player_stats[player_id]['mean_points'] + 
                             player_stats[player_id]['std_points'])
            })
            
            # Features de context
            X.update({
                'player_consistency': player_stats[player_id]['consistency'],
                'high_score_probability': player_stats[player_id]['high_score_freq'],
                'points_vs_average': recent_points[-1] - player_stats[player_id]['mean_points'],
                'current_form': np.mean(recent_points) - player_stats[player_id]['mean_points']
            })
            
            # Features de l'oponent
            opponent_id = target_game['opponent_team']
            opponent_games = df[df['opponent_team'] == opponent_id].tail(5)
            X.update({
                'opponent_goals_conceded_avg': opponent_games['goals_conceded'].mean(),
                'opponent_clean_sheets_ratio': opponent_games['clean_sheets'].mean(),
                'opponent_difficulty': calculate_opponent_difficulty(opponent_games)
            })
            
            feature_list.append(X)
            
            if len(player_info_list) == 0 or not any(p['player_id'] == player_id for p in player_info_list):
                player_info_list.append({
                    'player_id': player_id,
                    'player_name': target_game['player_name'],
                    'team': target_game['team'],
                    'consistency_score': player_stats[player_id]['consistency'],
                    'exceptional_rate': player_stats[player_id]['high_score_freq']
                })
    
    features_df = pd.DataFrame(feature_list)
    player_info_df = pd.DataFrame(player_info_list)
    
    return features_df, player_info_df

def calculate_opponent_difficulty(opponent_games):
    """
    Calcula un índex de dificultat de l'oponent basat en els seus últims partits.
    """
    return (opponent_games['goals_conceded'].mean() * 0.3 +
            (1 - opponent_games['clean_sheets'].mean()) * 0.3 +
            opponent_games['team_h_score'].mean() * 0.4)

def plot_temporal_results(y_true, y_pred, split_name='test'):
    """
    Crea un gràfic temporal de puntuacions reals vs predites només amb punts.
    """
    plt.figure(figsize=(15, 8))
    
    # Crear índex per l'eix x
    x = range(len(y_true))
    
    # Només els punts, sense línies
    plt.scatter(x, y_true, color='blue', alpha=0.5, s=20, label='Valors Reals')
    plt.scatter(x, y_pred, color='orange', alpha=0.5, s=20, label='Prediccions')
    
    plt.title('Resultats Temporals (Prediccions vs Reals)')
    plt.xlabel('Índex de Mostra')
    plt.ylabel('Punts')
    plt.grid(True, alpha=0.3)
    
    # Estadístiques
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    corr = np.corrcoef(y_true, y_pred)[0,1]
    
    stats_text = f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nCorrelació: {corr:.2f}'
    plt.text(0.02, 0.95, stats_text,
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='top')
    
    plt.legend()
    plt.tight_layout()
    return plt


    
if __name__ == "__main__":
    # Carregar dades
    print("Carregant dades...")
    df = pd.read_csv('./data/fantasy_data.csv')
    
    # Crear enhanced features
    print("\nCreant features avançades...")
    features_df, player_info_df = create_enhanced_features(df, n_prev_games=4)
    
    print(f"\nDimensió del dataset: {features_df.shape}")
    print(f"\nNoms de les columnes: {features_df.columns.tolist()}")
    print(f"Nombre de jugadors: {len(player_info_df)}")
    
    # Entrenar model i obtenir prediccions
    print("\nEntrenant model...")
    model, predictions, data_splits = train_and_predict(features_df)
    
    # Avaluar errors
    print("\nAvaluant resultats...")
    metrics = evaluate_errors(predictions, data_splits)
    
    # Anàlisi detallat dels resultats
    for split in ['val', 'test']:
        y_true = data_splits[f'y_{split}']
        y_pred = predictions[split]
        
        # Calcular errors per rang de punts
        print(f"\nAnàlisi detallat pel conjunt de {split}:")
        
        # Errors per rang de punts reals
        ranges = [(0,2), (2,5), (5,10), (10,float('inf'))]
        print("\nErrors per rang de punts:")
        for min_p, max_p in ranges:
            mask = (y_true >= min_p) & (y_true < max_p)
            if mask.any():
                mae = mean_absolute_error(y_true[mask], y_pred[mask])
                n_samples = mask.sum()
                print(f"Punts [{min_p}-{max_p}]: MAE = {mae:.2f} (n={n_samples})")
        
        # Anàlisi d'actuacions excepcionals
        threshold = y_true.mean() + y_true.std()
        exceptional_mask = y_true > threshold
        if exceptional_mask.any():
            mae_exceptional = mean_absolute_error(y_true[exceptional_mask], y_pred[exceptional_mask])
            n_exceptional = exceptional_mask.sum()
            print(f"\nActuacions excepcionals (>{threshold:.1f} punts):")
            print(f"MAE = {mae_exceptional:.2f} (n={n_exceptional})")
        
        # Estadístiques de predicció
        print(f"\nEstadístiques de predicció pel {split}:")
        print(f"Mitjana real: {y_true.mean():.2f}")
        print(f"Mitjana predita: {y_pred.mean():.2f}")
        print(f"Desviació estàndard real: {y_true.std():.2f}")
        print(f"Desviació estàndard predita: {y_pred.std():.2f}")
        
        # Percentatge d'errors per rang
        errors = np.abs(y_true - y_pred)
        print("\nDistribució d'errors:")
        for error_threshold in [1, 2, 3, 5]:
            pct = (errors <= error_threshold).mean() * 100
            print(f"Prediccions amb error ≤{error_threshold}: {pct:.1f}%")

            # Crear scatter plots
    for split in ['val', 'test']:
        y_true = data_splits[f'y_{split}']
        y_pred = predictions[split]
        
        # Crear i guardar scatter plot
        plt_scatter = plot_temporal_results(y_true, y_pred, split)
        plt_scatter.savefig(f'results/scatter_plot_{split}.png')
        plt.close()
            