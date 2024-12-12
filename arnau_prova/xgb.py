# train_model.py

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
    
if __name__ == "__main__":
    # Carregar dades
    df = pd.read_csv('./data/fantasy_data.csv')
    features_df, player_info_df = load_or_create_features(df, n_prev_games=2)
    
    # Entrenar model i obtenir prediccions
    model, predictions, data_splits = train_and_predict(features_df)
    
    # Avaluar errors
    metrics = evaluate_errors(predictions, data_splits)