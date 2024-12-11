# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 23:18:15 2024

@author: nildi
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

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


# Convertir target a categories
def convert_to_categories(df, target):
    # Definir intervals per als punts
    bins = [-1, 3, 7, float('inf')]  # Exemple: 0-3, 4-7, 8+
    labels = ['low', 'medium', 'high']
    df['target_class'] = pd.cut(df[target], bins=bins, labels=labels)
    return df

# Modificar funció d'entrenament per classificació
def train_classification_model(df, feature_columns, target_class):
    # Eliminar files amb valors nuls
    df_clean = df.dropna(subset=feature_columns + [target_class])
    
    # Dividir en features i target
    X = df_clean[feature_columns]
    y = df_clean[target_class]
    
    # Dividir en train i test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Crear i entrenar el model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Prediccions i matriu de confusió
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=['low', 'medium', 'high'])
    print("Matriu de confusió:")
    print(cm)
    print("\nInforme de classificació:")
    print(classification_report(y_test, y_pred))
    
    # Calcular i mostrar la corba ROC
    print("\nCalculant la corba ROC...")
    plot_roc_curve(model, X_test, y_test, classes=['low', 'medium', 'high'])
    
    return model



# Funció per calcular i mostrar la corba ROC
def plot_roc_curve(model, X_test, y_test, classes):
    # Predir probabilitats per a cada classe
    y_prob = model.predict_proba(X_test)
    
    # Binaritzar les etiquetes (necessari per a ROC multiclasse)
    y_test_binarized = label_binarize(y_test, classes=classes)
    
    # Calcular la corba ROC per a cada classe
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i, class_label in enumerate(classes):
        fpr[class_label], tpr[class_label], _ = roc_curve(y_test_binarized[:, i], y_prob[:, i])
        roc_auc[class_label] = auc(fpr[class_label], tpr[class_label])
    
    # Dibuixar la corba ROC
    plt.figure(figsize=(10, 8))
    for class_label in classes:
        plt.plot(fpr[class_label], tpr[class_label],
                 label=f"Classe {class_label} (AUC = {roc_auc[class_label]:.2f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label="Classificació aleatòria")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Taxa de falsos positius (FPR)")
    plt.ylabel("Taxa de vertaders positius (TPR)")
    plt.title("Corba ROC multiclasse")
    plt.legend(loc="lower right")
    plt.show()

# Modificar main
def main():
    # 1. Preparar les dades
    print("Carregant i preparant les dades...")
    df, feature_columns, target = prepare_data('data/fantasy_data.csv')
    
    # Convertir target a categories
    df = convert_to_categories(df, target)
    
    # 2. Entrenar el model de classificació
    print("\nEntrenant el model de classificació...")
    model = train_classification_model(df, feature_columns, 'target_class')
    
    # 3. Guardar el model
    print("\nGuardant el model...")
    joblib.dump(model, 'model/model_classification.pkl')

if __name__ == "__main__":
    main()