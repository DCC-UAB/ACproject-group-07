# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
from data_processing import load_or_create_features, get_test_player_names, get_player_info
import matplotlib.pyplot as plt
    


def plot_fold_results(fold_results):
    """
    Crea visualitzacions dels resultats per cada fold.
    """
   
    # Preparar dades
    folds = [r['fold'] for r in fold_results]
    maes = [r['mae'] for r in fold_results]
    rmses = [r['rmse'] for r in fold_results]
    
    # Crear figura amb dos subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot MAE
    ax1.bar(folds, maes, color='skyblue')
    ax1.axhline(y=np.mean(maes), color='red', linestyle='--', label=f'Mitjana: {np.mean(maes):.2f}')
    ax1.set_title('MAE per cada Fold')
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('MAE')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot RMSE
    ax2.bar(folds, rmses, color='lightgreen')
    ax2.axhline(y=np.mean(rmses), color='red', linestyle='--', label=f'Mitjana: {np.mean(rmses):.2f}')
    ax2.set_title('RMSE per cada Fold')
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('RMSE')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    return fig

def train_with_cross_validation(features_df, n_folds=5, random_state=42):
    """
    Entrena i avalua el model utilitzant k-fold cross validation.
    """
    # Preparar dades
    X = features_df.drop(columns=['player_id', 'target_total_points', 'prediction_game_number', 'prediction_date'])
    y = features_df['target_total_points']

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    mae_scores = []
    rmse_scores = []
    fold_results = []
    
    print(f"\nRealitzant {n_folds}-fold cross validation...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            early_stopping_rounds=10,
            eval_metric='mae'
        )
        
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        val_pred = model.predict(X_val)
        
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
    
    # Guardar resultats en CSV
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv('results/fold_results_2.csv', index=False)
    
    # Crear i guardar plots
    fig = plot_fold_results(fold_results)
    fig.savefig('results/fold_results_2.png')
    plt.close()
    
    print("\nResultats finals de la validació creuada:")
    print(f"MAE mitjà: {np.mean(mae_scores):.2f} (±{np.std(mae_scores):.2f})")
    print(f"RMSE mitjà: {np.mean(rmse_scores):.2f} (±{np.std(rmse_scores):.2f})")
    
    # Entrenar model final
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
    
    # Mostrar resultats detallats
    print("\nResultats detallats per cada fold:")
    for fold_result in cv_results['fold_results']:
        print(f"Fold {fold_result['fold']}:")
        print(f"  MAE: {fold_result['mae']:.2f}")
        print(f"  RMSE: {fold_result['rmse']:.2f}")