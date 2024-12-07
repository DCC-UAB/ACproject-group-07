
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

def prepare_data(df):
    """
    Prepara les dades pel model assegurant que totes les columnes necessàries existeixen
    """
    # Primer, imprimim les columnes disponibles per debug
    print("Columnes disponibles al dataset:")
    print(df.columns.tolist())
    
    # Fem una còpia per no modificar les dades originals
    df_model = df.copy()
    
    # Definim les columnes esperades
    required_columns = {
        'categorical': ['opponent_team', 'team', 'player_name'],
        'datetime': ['kickoff_time'],
        'numeric': ['minutes', 'goals_scored', 'assists', 'clean_sheets', 'goals_conceded',
                   'own_goals', 'penalties_saved', 'penalties_missed', 'yellow_cards',
                   'red_cards', 'saves', 'bonus', 'bps', 'influence', 'creativity',
                   'threat', 'ict_index', 'total_points']
    }
    
    # Verifiquem quines columnes falten
    missing_columns = []
    for category, cols in required_columns.items():
        for col in cols:
            if col not in df_model.columns:
                missing_columns.append(col)
    
    if missing_columns:
        print("\nATENCIÓ: Falten les següents columnes:")
        print(missing_columns)
        raise ValueError(f"Falten columnes necessàries al dataset: {missing_columns}")
    
    # Codifiquem variables categòriques
    le = LabelEncoder()
    for col in required_columns['categorical']:
        df_model[col] = le.fit_transform(df_model[col])
    
    # Convertim dates si la columna existeix
    if 'kickoff_time' in df_model.columns:
        df_model['kickoff_time'] = pd.to_datetime(df_model['kickoff_time'])
        df_model['month'] = df_model['kickoff_time'].dt.month
        df_model['day_of_week'] = df_model['kickoff_time'].dt.dayofweek
    
    # Gestionem valors nuls
    for col in required_columns['numeric']:
        df_model[col].fillna(0, inplace=True)
    
    return df_model

# Funció per descriure el dataset
def describe_dataset(df):
    """
    Mostra informació detallada sobre el dataset
    """
    print("\n=== DESCRIPCIÓ DEL DATASET ===")
    print(f"\nDimensiones: {df.shape}")
    print("\nColumnes disponibles:")
    for col in df.columns:
        print(f"- {col} ({df[col].dtype})")
    
    print("\nMostra de les primeres 5 files:")
    print(df.head())
    
    print("\nResum estadístic:")
    print(df.describe())
    
    print("\nValors nuls per columna:")
    null_counts = df.isnull().sum()
    print(null_counts[null_counts > 0])


def list_players_by_team(df):
    """
    Llista els equips disponibles i mostra els jugadors d'un equip seleccionat per l'usuari.
    """
    try:
        # Llistar equips únics
        print("\n=== EQUIPS DISPONIBLES ===")
        available_teams = sorted(df['team'].unique())
        print(", ".join(available_teams))
        
        # Demanar a l'usuari que triï un equip
        team_name = input("\nIntrodueix el nom de l'equip (exactament com apareix a la llista): ")
        
        if team_name not in available_teams:
            print(f"\nError: L'equip '{team_name}' no es troba al dataset. Si us plau, tria un equip de la llista.")
            return
        
        # Filtrar els jugadors de l'equip seleccionat
        team_players = df[df['team'] == team_name]['player_name'].unique()
        print(f"\n=== JUGADORS DE L'EQUIP '{team_name}' ===")
        for player in sorted(team_players):
            print(f"- {player}")
    
    except Exception as e:
        print(f"\nError en obtenir la llista de jugadors: {str(e)}")

def main():
    """
    Menú principal per interactuar amb el dataset.
    L'usuari pot triar entre descriure el dataset o veure els jugadors d'un equip.
    """
    try:
        # Carregar el dataset
        df = pd.read_csv('data/fantasy_data.csv')

        while True:
            # Mostrar el menú
            print("\n=== MENÚ PRINCIPAL ===")
            print("1. Descriure el dataset")
            print("2. Veure jugadors d'un equip")
            print("3. Sortir")
            
            # Demanar l'opció a l'usuari
            option = input("\nTria una opció (1, 2 o 3): ")

            if option == '1':
                # Descriure el dataset
                describe_dataset(df)
            elif option == '2':
                # Llistar jugadors d'un equip
                list_players_by_team(df)
            elif option == '3':
                # Sortir del programa
                print("\nSortint del programa. Adéu!")
                break
            else:
                print("\nOpció no vàlida. Si us plau, tria una opció vàlida.")
    except FileNotFoundError:
        print("\nError: No s'ha trobat el fitxer 'data/fantasy_data_cleaned.csv'.")
        print("Si us plau, verifica que el fitxer existeix i està al directori correcte.")
    except Exception as e:
        print(f"\nError inesperat: {str(e)}")

if __name__ == "__main__":
    main()
