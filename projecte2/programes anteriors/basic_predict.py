import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# Funció per crear features històriques
def create_historical_features(df, player_id, n_matches=3):
    # Ordenem per jugador i data
    df = df.sort_values(['player_id', 'kickoff_time'])
    
    # Creem features amb mitjanes mòbils
    historical_features = []
    for stat in ['minutes', 'total_points', 'goals_scored', 'assists']:
        df[f'last_{n_matches}_{stat}'] = df.groupby('player_id')[stat].transform(
            lambda x: x.rolling(n_matches, min_periods=1).mean()
        )
        historical_features.append(f'last_{n_matches}_{stat}')
    
    return df, historical_features

# Carregar i preparar les dades
def prepare_data(file_path):
    # Llegir el CSV
    df = pd.read_csv(file_path)
    
    # Convertir kickoff_time a datetime
    df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])
    
    # Codificar variables categòriques
    le = LabelEncoder()
    df['team_encoded'] = le.fit_transform(df['team'])
    df['opponent_team_encoded'] = le.fit_transform(df['opponent_team'])
    
    # Crear features històriques
    df, historical_features = create_historical_features(df, 'player_id')
    
    # Seleccionar features per al model
    feature_columns = [
        'team_encoded',
        'opponent_team_encoded',
        'was_home',
        'value',
        'minutes',
        'goals_scored',
        'assists',
        'clean_sheets'
    ] + historical_features
    
    target = 'total_points'
    
    return df, feature_columns, target

# Entrenar el model
def train_model(df, feature_columns, target):
    # Eliminar files amb valors nuls
    df_clean = df.dropna(subset=feature_columns + [target])
    
    # Dividir en features i target
    X = df_clean[feature_columns]
    y = df_clean[target]
    
    # Dividir en train i test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Crear i entrenar el model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Avaluar el model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"R² Score (Train): {train_score:.3f}")
    print(f"R² Score (Test): {test_score:.3f}")
    
    return model, X_test, y_test

# Funció per fer prediccions per la següent jornada
def predict_next_gameweek(model, df, player_name, feature_columns):
    # Obtenir les dades del jugador
    player_data = df[df['player_name'] == player_name].iloc[-1:]
    
    if player_data.empty:
        raise ValueError(f"No s'ha trobat el jugador: {player_name}")
    
    # Assegurar que tenim totes les features necessàries
    player_features = player_data[feature_columns]
    
    # Fer la predicció
    prediction = model.predict(player_features)
    
    return prediction[0], player_data

def main():
    # 1. Preparar les dades
    print("Carregant i preparant les dades...")
    df, feature_columns, target = prepare_data('data/fantasy_data.csv')
    
    # 2. Entrenar el model
    print("\nEntrenant el model...")
    model, X_test, y_test = train_model(df, feature_columns, target)
    
    # 3. Guardar el model
    print("\nGuardant el model...")
    joblib.dump(model, 'model/model.pkl')
    
    # 4. Demanar el nom del jugador i fer la predicció
    player_name = input("\nIntrodueix el nom del jugador: ")
    try:
        prediction, player_data = predict_next_gameweek(model, df, player_name, feature_columns)
        print(f"\nPredicció per {player_name}:")
        print(f"Equip: {player_data['team'].values[0]}")
        print(f"Valor: {player_data['value'].values[0]}")
        print(f"Predicció de punts: {prediction:.2f}")
    except ValueError as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()