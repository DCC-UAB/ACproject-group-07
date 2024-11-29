import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
from typing import List, Tuple
import os


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Carrega i neteja les dades
    """
    df = pd.read_csv(file_path)
    
    # Convertir dates
    df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])
    
    # Assegurar que tenim totes les columnes necessàries
    required_columns = [
        'player_name', 'team', 'opponent_team', 'was_home',
        'minutes', 'goals_scored', 'assists', 'clean_sheets', 'saves',
        'bonus', 'bps', 'influence', 'creativity', 'threat', 'ict_index',
        'total_points', 'value'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Falten les següents columnes: {missing_columns}")
    
    return df

def create_historical_features(df: pd.DataFrame, n_matches: int = 5) -> Tuple[pd.DataFrame, List[str]]:
    """
    Crea features històriques
    """
    df = df.sort_values(['player_id', 'kickoff_time'])
    
    historical_features = []
    stats_to_track = [
        'minutes', 'total_points', 'goals_scored', 'assists', 'clean_sheets',
        'bonus', 'bps', 'influence', 'creativity', 'threat', 'ict_index'
    ]
    
    for stat in stats_to_track:
        col_name = f'last_{n_matches}_{stat}'
        df[col_name] = df.groupby('player_id')[stat].transform(
            lambda x: x.rolling(n_matches, min_periods=1).mean()
        )
        historical_features.append(col_name)
    
    return df, historical_features

def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], str]:
    """
    Prepara les dades pel model
    """
    # Codificar variables categòriques
    le_team = LabelEncoder()
    df['team_encoded'] = le_team.fit_transform(df['team'])
    df['opponent_team_encoded'] = le_team.transform(df['opponent_team'])
    
    # Crear features històriques
    df, historical_features = create_historical_features(df)
    
    # Seleccionar features pel model
    feature_columns = [
        'team_encoded',
        'opponent_team_encoded',
        'was_home',
        'value',
        'minutes',
        'goals_scored',
        'assists',
        'clean_sheets',
        'bonus',
        'bps',
        'influence',
        'creativity',
        'threat',
        'ict_index'
    ] + historical_features
    
    target = 'total_points'
    
    return df, feature_columns, target

def train_model(df: pd.DataFrame, feature_columns: List[str], target: str) -> RandomForestRegressor:
    """
    Entrena el model
    """
    df_clean = df.dropna(subset=feature_columns + [target])
    X = df_clean[feature_columns]
    y = df_clean[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Mostrar mètriques
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"\nMètriques del model:")
    print(f"R² Score (Train): {train_score:.3f}")
    print(f"R² Score (Test): {test_score:.3f}")
    
    return model

def predict_team_performance(
    df: pd.DataFrame,
    team_name: str,
    opponent_team: str,
    is_home: bool,
    model: RandomForestRegressor,
    feature_columns: List[str]
) -> pd.DataFrame:
    """
    Prediu el rendiment de tot l'equip
    """
    # Verificar que l'equip existeix
    if team_name not in df['team'].unique():
        available_teams = sorted(df['team'].unique())
        raise ValueError(
            f"Equip '{team_name}' no trobat. Equips disponibles:\n{', '.join(available_teams)}"
        )
    
    # Verificar que l'equip rival existeix
    if opponent_team not in df['team'].unique():
        raise ValueError(f"Equip rival '{opponent_team}' no trobat")
    
    # Filtrar jugadors de l'equip
    team_players = df[df['team'] == team_name]
    
    # Obtenir últimes dades per cada jugador
    latest_records = team_players.groupby('player_name').last().reset_index()
    
    # Preparar dades per la predicció
    latest_records['opponent_team'] = opponent_team
    latest_records['was_home'] = is_home
    
    # Actualitzar encodings
    le = LabelEncoder()
    le.fit(df['team'])
    latest_records['opponent_team_encoded'] = le.transform([opponent_team] * len(latest_records))
    
    # Fer prediccions
    predictions = model.predict(latest_records[feature_columns])
    
    # Crear DataFrame amb resultats
    results = pd.DataFrame({
        'player_name': latest_records['player_name'],
        'predicted_points': predictions,
        'value': latest_records['value'],
        'minutes_played': latest_records['minutes'],
        'influence': latest_records['influence'],
        'creativity': latest_records['creativity'],
        'threat': latest_records['threat'],
        'ict_index': latest_records['ict_index'],
        'goals_last_5': latest_records['last_5_goals_scored'],
        'assists_last_5': latest_records['last_5_assists']
    })
    
    return results

def get_best_eleven(players_df: pd.DataFrame) -> pd.DataFrame:
    """
    Selecciona els 11 millors jugadors basats en la puntuació prevista
    """
    # Filtrar jugadors que han jugat regularment (més de 45 minuts de mitjana)
    active_players = players_df[players_df['minutes_played'] > 45]
    
    # Seleccionar els 11 millors
    best_eleven = active_players.nlargest(11, 'predicted_points')
    
    return best_eleven

def display_results(
    best_eleven: pd.DataFrame,
    team_name: str,
    opponent_team: str,
    is_home: bool
) -> None:
    """
    Mostra els resultats de forma organitzada
    """
    location = 'casa' if is_home else 'fora'
    
    print(f"\nMillors 11 jugadors per {team_name} contra {opponent_team}")
    print(f"Partit a: {location}")
    print("-" * 100)
    print(f"{'Jugador':<25} {'Punts':>8} {'Valor':>8} {'ICT':>8} {'Infl':>8} {'Creat':>8} {'Threat':>8} {'G/A 5':>8}")
    print("-" * 100)
    
    for _, player in best_eleven.iterrows():
        g_a_5 = f"{player['goals_last_5']:.1f}/{player['assists_last_5']:.1f}"
        print(
            f"{player['player_name']:<25} "
            f"{player['predicted_points']:>8.1f} "
            f"{player['value']:>8.1f} "
            f"{player['ict_index']:>8.1f} "
            f"{player['influence']:>8.1f} "
            f"{player['creativity']:>8.1f} "
            f"{player['threat']:>8.1f} "
            f"{g_a_5:>8}"
        )
    
    print("\nEstadístiques de l'equip:")
    print(f"Puntuació total prevista: {best_eleven['predicted_points'].sum():.1f}")
    print(f"Valor total de l'equip: {best_eleven['value'].sum():.1f}M")
    print(f"ICT Index mitjà: {best_eleven['ict_index'].mean():.1f}")
    print(f"Gols últims 5 partits: {best_eleven['goals_last_5'].sum():.1f}")
    print(f"Assistències últims 5 partits: {best_eleven['assists_last_5'].sum():.1f}")



def main():
    model_path = 'model/model_team.pkl'

    # 1. Comprovar si el model ja està guardat
    if os.path.exists(model_path):
        print("\nCarregant el model existent...")
        model = joblib.load(model_path)

        # Preparar les dades per al model carregat
        df = load_and_clean_data('data/fantasy_data.csv')
        df, feature_columns, _ = prepare_data(df)

    else:
        print("\nCarregant i preparant les dades...")
        df = load_and_clean_data('data/fantasy_data.csv')
        df, feature_columns, target = prepare_data(df)

        print("\nEntrenant el model...")
        model = train_model(df, feature_columns, target)

        # Crear la carpeta si no existeix i guardar el model
        os.makedirs('model', exist_ok=True)
        print("\nGuardant el model...")
        joblib.dump(model, model_path)

    # 2. Mostrar equips disponibles
    available_teams = sorted(df['team'].unique())
    print("\nEquips disponibles:")
    print(", ".join(available_teams))

    while True:
        # 3. Demanar inputs a l'usuari
        team_name = input("\nIntrodueix l'equip (o 'q' per sortir): ")
        if team_name.lower() == 'q':
            break

        if team_name not in available_teams:
            print(f"Equip no vàlid. Escull entre: {', '.join(available_teams)}")
            continue

        opponent_team = input("Contra quin equip juga? ")
        if opponent_team not in available_teams:
            print(f"Equip rival no vàlid. Escull entre: {', '.join(available_teams)}")
            continue

        while True:
            location = input("Juga a casa (C) o fora (F)? ").upper()
            if location in ['C', 'F']:
                break
            print("Si us plau, introdueix 'C' per casa o 'F' per fora.")

        is_home = location == 'C'

        try:
            # 4. Obtenir prediccions
            team_predictions = predict_team_performance(
                df, team_name, opponent_team, is_home, model, feature_columns
            )

            # 5. Seleccionar els millors 11 jugadors
            best_eleven = get_best_eleven(team_predictions)

            # 6. Mostrar resultats
            display_results(best_eleven, team_name, opponent_team, is_home)

        except ValueError as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()

