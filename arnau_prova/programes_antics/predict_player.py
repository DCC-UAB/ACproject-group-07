import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import joblib
import os

# Carregar les dades des de l'arxiu CSV
df = pd.read_csv('./data/fantasy_data.csv')

# Preparar les dades
df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])
df['round'] = df['kickoff_time'].dt.strftime('%Y-%m-%d')

def assign_round_numbers(df):
    """Assigna números de jornada (1-18) per cada meitat de temporada"""
    df = df.sort_values(by=['player_id', 'kickoff_time'])
    
    # Crear números de jornada originals (1-36)
    df['original_round'] = df.groupby('player_id').cumcount() + 1
    
    # Crear números de jornada normalitzats (1-18)
    df['round_number'] = df['original_round'].apply(lambda x: x if x <= 18 else x - 18)
    
    return df

df = assign_round_numbers(df)

# Dividir en train i test (primera i segona meitat de temporada)
train_df = df[df['original_round'] <= 18].copy()
test_df = df[df['original_round'] > 18].copy()

# Característiques per al model
feature_columns = [
    'assists', 'bonus', 'bps', 'clean_sheets', 'creativity',
    'goals_conceded', 'goals_scored', 'ict_index', 'influence',
    'minutes', 'own_goals', 'red_cards', 'saves', 'selected', 'team_a_score', 'team_h_score',
    'threat', 'transfers_balance', 'transfers_in', 'transfers_out',
    'value', 'was_home'
]

X_train = train_df[feature_columns]
y_train = train_df['total_points']
X_test = test_df[feature_columns]
y_test = test_df['total_points']

# Comprovar si el model existeix
model_path = 'fantasy_model.joblib'
if os.path.exists(model_path):
    print("Carregant model existent...")
    model = joblib.load(model_path)
else:
    print("Entrenant nou model...")
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Guardar el model
    print("Guardant el model...")
    joblib.dump(model, model_path)
    print("Model guardat correctament")

# Funcionalitat per predir punts
while True:
    print("\nJornades disponibles: 1-18")
    round_number = int(input("Introdueix el número de la jornada (1-18) que vols predir: "))
    
    if round_number < 1 or round_number > 18:
        print("Jornada no vàlida. Ha d'estar entre 1 i 18.")
        continue

    player_name = input("Introdueix el nom del jugador (o 'q' per sortir): ")
    if player_name.lower() == 'q':
        break
        
    try:
        # Buscar el jugador en el test set (jornades 19-36)
        player_data = test_df[test_df['player_name'].str.contains(player_name, case=False, na=False)]
        
        if player_data.empty:
            print("No s'ha trobat dades per al jugador especificat.")
            continue
            
        # Buscar la jornada equivalent en el test set
        X_player = player_data[player_data['round_number'] == round_number][feature_columns]
        
        if X_player.empty:
            print(f"No hi ha dades per la jornada {round_number} per aquest jugador.")
            continue
            
        # Fer predicció
        predicted_points = model.predict(X_player)
        real_points = player_data[player_data['round_number'] == round_number]['total_points'].values[0]
        real_round = player_data[player_data['round_number'] == round_number]['original_round'].values[0]
        
        print(f"\nPredicció per al jugador {player_name}:")
        print(f"Jornada predita: {round_number} (Jornada real: {real_round})")
        print(f"Punts predits: {predicted_points[0]:.2f}")
        print(f"Punts reals: {real_points}")
        
    except Exception as e:
        print(f"Error: {e}")