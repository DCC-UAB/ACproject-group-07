import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt

def plot_game_errors(stats_df):
    """
    Crea visualitzacions dels errors per número de partit.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot de l'error mitjà
    ax1.plot(stats_df['game_number'], stats_df['mean_error'], 'b-', label='Error mitjà')
    ax1.fill_between(
        stats_df['game_number'],
        stats_df['mean_error'] - stats_df['std_error'],
        stats_df['mean_error'] + stats_df['std_error'],
        alpha=0.2,
        color='blue',
        label='Desviació estàndard'
    )
    ax1.set_title('Error mitjà per número de partit')
    ax1.set_xlabel('Número de partit')
    ax1.set_ylabel('Error mitjà')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot del nombre de mostres
    ax2.bar(stats_df['game_number'], stats_df['count'], alpha=0.7, color='lightgreen')
    ax2.set_title('Nombre de prediccions per partit')
    ax2.set_xlabel('Número de partit')
    ax2.set_ylabel('Nombre de prediccions')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def analyze_error_by_game_number(features_df, n_splits=5, random_state=42):
    """
    Analitza l'error per cada número de partit.
    """
    # Preparar dades
    X = features_df.drop(columns=['player_id', 'target_total_points', 'prediction_game_number', 'prediction_date'])
    y = features_df['target_total_points']
    game_numbers = features_df['prediction_game_number']
    
    # Configurar la validació creuada
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Diccionari per guardar errors per número de partit
    game_errors = {}
    
    print(f"\nRealitzant anàlisi amb {n_splits}-fold cross validation...")
    
    # Per cada fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"\nProcessant fold {fold}...")
        
        # Split de dades
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        games_val = game_numbers.iloc[val_idx]
        
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
        
        # Calcular errors per cada partit
        for pred, true, game in zip(val_pred, y_val, games_val):
            error = abs(pred - true)
            if game not in game_errors:
                game_errors[game] = []
            game_errors[game].append(error)
    
    # Calcular estadístiques per cada partit
    game_stats = []
    for game in sorted(game_errors.keys()):
        errors = game_errors[game]
        game_stats.append({
            'game_number': game,
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'median_error': np.median(errors),
            'count': len(errors)
        })
    
    # Convertir a DataFrame i guardar
    stats_df = pd.DataFrame(game_stats)
    stats_df.to_csv('results/rfg_error_by_game_2.csv', index=False)
    
    # Crear i guardar visualització
    fig = plot_game_errors(stats_df)
    fig.savefig('results/rfg_error_by_game_2.png')
    plt.close()
    
    # Mostrar estadístiques generals
    print("\nEstadístiques globals:")
    print(f"Error mitjà total: {stats_df['mean_error'].mean():.2f}")
    print(f"Desviació estàndard mitjana: {stats_df['std_error'].mean():.2f}")
    print(f"Nombre total de partits analitzats: {len(stats_df)}")
    
    return stats_df

if __name__ == "__main__":
    # Carregar dades
    features_df = pd.read_csv('data/features_df_2.csv')
    
    # Analitzar errors per partit
    stats_df = analyze_error_by_game_number(features_df)
    
    # Mostrar alguns resultats
    print("\nResum d'errors per partit:")
    print(stats_df.to_string(index=False))