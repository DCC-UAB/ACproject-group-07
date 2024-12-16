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
      

def plot_predictions_vs_real(y_true, y_pred, split_name='test'):
    """
    Crea un gràfic temporal de puntuacions reals vs predites.
    """
    plt.figure(figsize=(15, 8))
    
    # Crear índex per l'eix x
    x = range(len(y_true))
    
    # Gràfic amb línies i punts
    plt.scatter(x, y_true, color='blue', alpha=0.5, s=20)
    plt.scatter(x, y_pred, color='orange', alpha=0.5, s=20)
    
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



def analyze_errors_by_range(y_true, y_pred, split_name='test'):
    """
    Analitza els errors per diferents rangs de puntuació.
    """
    print(f"\nAnàlisi detallat pel conjunt de {split_name}:")
    
    # Errors per rang de punts
    ranges = [(0,2), (2,5), (5,10), (10,float('inf'))]
    print("\nErrors per rang de punts:")
    for min_p, max_p in ranges:
        mask = (y_true >= min_p) & (y_true < max_p)
        if mask.any():
            mae = mean_absolute_error(y_true[mask], y_pred[mask])
            n_samples = mask.sum()
            print(f"Punts [{min_p}-{max_p}]: MAE = {mae:.2f} (n={n_samples})")
    
    # Actuacions excepcionals
    threshold = y_true.mean() + y_true.std()
    exceptional_mask = y_true > threshold
    if exceptional_mask.any():
        mae_exceptional = mean_absolute_error(y_true[exceptional_mask], y_pred[exceptional_mask])
        n_exceptional = exceptional_mask.sum()
        print(f"\nActuacions excepcionals (>{threshold:.1f} punts):")
        print(f"MAE = {mae_exceptional:.2f} (n={n_exceptional})")
    
    # Estadístiques de predicció
    print(f"\nEstadístiques de predicció:")
    print(f"Mitjana real: {y_true.mean():.2f}")
    print(f"Mitjana predita: {y_pred.mean():.2f}")
    print(f"Desviació estàndard real: {y_true.std():.2f}")
    print(f"Desviació estàndard predita: {y_pred.std():.2f}")
    
    # Distribució d'errors
    errors = np.abs(y_true - y_pred)
    print("\nDistribució d'errors:")
    for error_threshold in [1, 2, 3, 5]:
        pct = (errors <= error_threshold).mean() * 100
        print(f"Prediccions amb error ≤{error_threshold}: {pct:.1f}%")

if __name__ == "__main__":
    # Carregar dades
    df = pd.read_csv('./data/fantasy_data.csv')
    features_df, player_info_df = load_or_create_features(df, n_prev_games=2)
    
    # Entrenar model i obtenir prediccions
    model, predictions, data_splits = train_and_predict(features_df)
    
    # Avaluar errors globals
    metrics = evaluate_errors(predictions, data_splits)
    
    # Analitzar errors per rang i crear gràfics temporals
    for split in ['val', 'test']:
        # Anàlisi per rangs
        analyze_errors_by_range(data_splits[f'y_{split}'], predictions[split], split)
        
        # Crear i guardar gràfic temporal
        plt_temporal = plot_predictions_vs_real(data_splits[f'y_{split}'], predictions[split], split)
        plt_temporal.savefig(f'results/temporal_plot_{split}.png', dpi=300, bbox_inches='tight')
        plt.close()
