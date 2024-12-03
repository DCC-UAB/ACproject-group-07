# Importació de llibreries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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
    features = []
    target = []

    for player_id in df['player_id'].unique():
        player_data = df[df['player_id'] == player_id]

        for i in range(n_prev_games, len(player_data)):
            last_games = player_data.iloc[i-n_prev_games:i]
            X = last_games[['assists', 'bonus', 'bps', 'clean_sheets', 'creativity', 'goals_conceded',
                            'goals_scored', 'ict_index', 'influence', 'minutes', 
                            'own_goals', 'penalties_missed', 'penalties_saved', 'red_cards', 'saves', 
                            'selected', 'team_a_score', 'team_h_score', 'threat', 'transfers_balance', 
                            'transfers_in', 'transfers_out', 'value', 'was_home', 'yellow_cards']].mean(axis=0)
            
            # Afegir player_id, player_name i team a les característiques
            X['player_id'] = player_id
            X['player_name'] = player_data.iloc[i]['player_name']
            X['team'] = player_data.iloc[i]['team']
            
            target_value = player_data.iloc[i]['total_points']
            features.append(X)
            target.append(target_value)
    
    return pd.DataFrame(features), target

# Inicialitzar paràmetres
n_prev_games = 5  # Fixem el nombre de partits utilitzats per les característiques
max_future_games = 5  # Nombre màxim de partits futurs a predir
error_per_step = []

# Crear les característiques i la variable dependent (puntuació)
X, y = create_features(df, n_prev_games)
X = pd.get_dummies(X, columns=['player_name', 'team'], drop_first=True)

# Dividir les dades en entrenament i prova
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el model de Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Prediccions iteratives
X_iter = X_test.copy()
y_iter = y_test.copy()

for step in range(1, max_future_games + 1):
    print(f"Predicció iterativa: Pas {step}")

    # Predir els valors per al conjunt iteratiu actual
    y_pred = rf.predict(X_iter)

    # Calcular mètriques d'error
    mae = mean_absolute_error(y_iter, y_pred)
    rmse = mean_squared_error(y_iter, y_pred, squared=False)

    # Guardar l'error d'aquest pas
    error_per_step.append({'step': step, 'MAE': mae, 'RMSE': rmse})

    # Simular la següent iteració: substituir "y_iter" per les prediccions
    # NOTA: No afegim `total_points` a `X_iter`, només actualitzem `y_iter`
    y_iter = y_pred  # Actualitzar les puntuacions a les prediccions anteriors

# Convertir els resultats en DataFrame
errors_df = pd.DataFrame(error_per_step)

# Mostrar resultats
print("\nEvolució de l'error per cada pas de predicció iterativa:")
print(errors_df)

# Gràfic de l'evolució dels errors
plt.figure(figsize=(10, 6))
plt.plot(errors_df['step'], errors_df['MAE'], marker='o', label='MAE')
plt.plot(errors_df['step'], errors_df['RMSE'], marker='o', label='RMSE')
plt.xlabel('Pas de predicció iterativa (nombre de partits predits)')
plt.ylabel('Error')
plt.title('Evolució de l\'error en prediccions iteratives')
plt.legend()
plt.grid()
plt.show()
