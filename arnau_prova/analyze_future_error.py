import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from create_features_future import create_features

def analyze_prediction_horizon(df, n_prev_games=2, max_future=6, n_folds=5, random_state=42):
    """
    Analitza com varia l'error segons quants partits endavant volem predir.
    """
    future_results = []
    
    # Per cada horitzó de predicció
    for n_future in range(1, max_future + 1):
        print(f"\nAnalitzant prediccions a {n_future} partits vista...")
        
        # Crear features per aquest horitzó
        features_df, _ = create_features(df, n_prev_games, n_future)
        
        # Preparar dades
        X = features_df.drop(columns=['player_id', 'target_total_points', 'prediction_game_number', 'prediction_date'])
        y = features_df['target_total_points']
        
        # Cross validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Entrenar model
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=random_state
            )
            
            model.fit(X_train, y_train)
            
            # Prediccions
            val_pred = model.predict(X_val)
            
            # Calcular errors
            mae = mean_absolute_error(y_val, val_pred)
            rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            
            fold_scores.append({
                'fold': fold,
                'mae': mae,
                'rmse': rmse
            })
        
        # Calcular mitjanes dels folds
        mean_mae = np.mean([score['mae'] for score in fold_scores])
        std_mae = np.std([score['mae'] for score in fold_scores])
        mean_rmse = np.mean([score['rmse'] for score in fold_scores])
        std_rmse = np.std([score['rmse'] for score in fold_scores])
        
        future_results.append({
            'n_future': n_future,
            'mean_mae': mean_mae,
            'std_mae': std_mae,
            'mean_rmse': mean_rmse,
            'std_rmse': std_rmse
        })
        
        print(f"MAE mitjà: {mean_mae:.2f} (±{std_mae:.2f})")
        print(f"RMSE mitjà: {mean_rmse:.2f} (±{std_rmse:.2f})")
    
    # Convertir a DataFrame i guardar
    results_df = pd.DataFrame(future_results)
    results_df.to_csv('results/future_horizon_analysis.csv', index=False)
    
    # Crear visualitzacions
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot MAE
    ax1.errorbar(
        results_df['n_future'], 
        results_df['mean_mae'],
        yerr=results_df['std_mae'],
        fmt='o-',
        capsize=5,
        color='blue',
        label='MAE'
    )
    ax1.set_title('Error Absolut Mitjà segons Horitzó de Predicció')
    ax1.set_xlabel('Partits endavant')
    ax1.set_ylabel('MAE')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot RMSE
    ax2.errorbar(
        results_df['n_future'], 
        results_df['mean_rmse'],
        yerr=results_df['std_rmse'],
        fmt='o-',
        capsize=5,
        color='green',
        label='RMSE'
    )
    ax2.set_title('Error Quadràtic Mitjà segons Horitzó de Predicció')
    ax2.set_xlabel('Partits endavant')
    ax2.set_ylabel('RMSE')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('results/future_horizon_analysis.png')
    plt.close()
    
    return results_df

if __name__ == "__main__":
    # Carregar dades
    df = pd.read_csv('./data/fantasy_data.csv')
    
    # Analitzar error per diferents horitzons de predicció
    results = analyze_prediction_horizon(
        df,
        n_prev_games=2,
        max_future=5,  # Analitzar fins a 5 partits endavant
        n_folds=5
    )
    
    # Mostrar resultats
    print("\nResum dels resultats:")
    print(results.to_string(index=False))