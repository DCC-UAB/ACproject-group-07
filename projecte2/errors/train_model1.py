import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Funció per extreure característiques temporals
def add_time_features(df):
    df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])
    df['kickoff_month'] = df['kickoff_time'].dt.month
    df['kickoff_week'] = df['kickoff_time'].dt.isocalendar().week
    df['kickoff_day'] = df['kickoff_time'].dt.day
    df['kickoff_weekday'] = df['kickoff_time'].dt.weekday
    df = df.sort_values(['player_id', 'kickoff_time'])
    df['days_since_last_match'] = df.groupby('player_id')['kickoff_time'].diff().dt.days.fillna(0)
    return df

# Funció per preparar les dades
def prepare_data_with_time_fixed_full(df):
    df = add_time_features(df)
    df = df.drop(columns=['kickoff_time'], errors='ignore')
    categorical_columns = df.select_dtypes(include=['object', 'bool']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    df_encoded = df_encoded.fillna(df_encoded.mean(numeric_only=True))
    target = 'total_points'
    feature_columns = df_encoded.columns.drop(target)
    return df_encoded, feature_columns, target

# Funció per entrenar el model i calcular mètriques
def train_model_with_metrics(df, feature_columns, target):
    df_clean = df.dropna(subset=[target])
    X = df_clean[feature_columns]
    y = df_clean[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return {
        "train_score": train_score,
        "test_score": test_score,
        "mae": mae,
        "rmse": rmse,
        "training_time": training_time
    }, model

# Funció per guardar els resultats en un fitxer .txt
def save_metrics_to_file(metrics, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        file.write("Mètriques del model:\n")
        file.write(f"R² Score (Train): {metrics['train_score']:.3f}\n")
        file.write(f"R² Score (Test): {metrics['test_score']:.3f}\n")
        file.write(f"MAE (Test): {metrics['mae']:.3f}\n")
        file.write(f"RMSE (Test): {metrics['rmse']:.3f}\n")
        file.write(f"Temps d'entrenament: {metrics['training_time']:.3f} segons\n")

# Main
if __name__ == "__main__":
    # Llegir les dades
    file_path = "data/fantasy_data.csv"  # Actualitza la ruta si cal
    df = pd.read_csv(file_path)

    # Preparar les dades amb kickoff_time
    print("Preparant les dades...")
    df_cleaned, feature_columns, target = prepare_data_with_time_fixed_full(df)

    # Entrenar el model
    print("Entrenant el model...")
    metrics, model = train_model_with_metrics(df_cleaned, feature_columns, target)

    # Mostrar les mètriques
    print("\nMètriques del model:")
    print(f"R² Score (Train): {metrics['train_score']:.3f}")
    print(f"R² Score (Test): {metrics['test_score']:.3f}")
    print(f"MAE (Test): {metrics['mae']:.3f}")
    print(f"RMSE (Test): {metrics['rmse']:.3f}")
    print(f"Temps d'entrenament: {metrics['training_time']:.3f} segons")

    # Guardar les mètriques en un fitxer .txt
    output_file = "errors/model_metrics.txt"
    save_metrics_to_file(metrics, output_file)
    print(f"\nLes mètriques s'han guardat a: {output_file}")
