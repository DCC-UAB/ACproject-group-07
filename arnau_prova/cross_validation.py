# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
from data_processing import load_or_create_features, get_test_player_names, get_player_info

def train_with_cross_validation(features_df, n_folds=5, random_state=42):
    """
    Entrena i avalua el model utilitzant k-fold cross validation.
    """
    # Preparar dades
    X = features_df.drop(columns=['player_id', 'target_total_points', 'prediction_game_number', 'prediction_date'])
    y = features_df['target_total_points']

    # Configurar la validació creuada
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # Llistes per guardar els resultats
    mae_scores = []
    rmse_scores = []
    fold_results = []
    
    print(f"\nRealitzant {n_folds}-fold cross validation...")
    
    # Per cada fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        # Split de dades per aquest fold
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Entrenar model
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            early_stopping_rounds=10,
            eval_metric='mae'
        )
        
        # Fit del model
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Prediccions
        val_pred = model.predict(X_val)
        
        # Calcular mètriques
        mae = mean_absolute_error(y_val, val_pred)
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        
        mae_scores.append(mae)
        rmse_scores.append(rmse)
        
        fold_results.append({
            'fold': fold,
            'mae': mae,
            'rmse': rmse
        })
        
        print(f"\nFold {fold}:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
    
    # Calcular i mostrar resultats finals
    print("\nResultats finals de la validació creuada:")
    print(f"MAE mitjà: {np.mean(mae_scores):.2f} (±{np.std(mae_scores):.2f})")
    print(f"RMSE mitjà: {np.mean(rmse_scores):.2f} (±{np.std(rmse_scores):.2f})")
    
    # Entrenar model final amb totes les dades
    print("\nEntrenant model final amb totes les dades...")
    final_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=random_state
    )
    final_model.fit(X, y)
    
    return final_model, {
        'fold_results': fold_results,
        'mean_mae': np.mean(mae_scores),
        'std_mae': np.std(mae_scores),
        'mean_rmse': np.mean(rmse_scores),
        'std_rmse': np.std(rmse_scores)
    }

if __name__ == "__main__":
    # Carregar dades
    df = pd.read_csv('./data/fantasy_data.csv')
    features_df, player_info_df = load_or_create_features(df, n_prev_games=2)
    
    # Entrenar amb cross validation
    model, cv_results = train_with_cross_validation(features_df, n_folds=5)
    
    # Mostrar resultats detallats per cada fold
    print("\nResultats detallats per cada fold:")
    for fold_result in cv_results['fold_results']:
        print(f"Fold {fold_result['fold']}:")
        print(f"  MAE: {fold_result['mae']:.2f}")
        print(f"  RMSE: {fold_result['rmse']:.2f}")