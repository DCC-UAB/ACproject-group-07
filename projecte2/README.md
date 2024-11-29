Projecte de Predicció en Fantasy Football

Aquest projecte inclou tres scripts Python que implementen models de Machine Learning per predir rendiments de jugadors i equips de futbol. Els scripts fan ús de dades històriques per crear prediccions personalitzades.

## Estructura del Projecte

### 1. `basic_predict.py`

#### **Funcionalitat**
Aquest script permet:
- Preparar dades històriques per a jugadors.
- Entrenar un model de Random Forest per predir els punts totals d'un jugador en un partit.
- Realitzar prediccions per un jugador específic basant-se en el seu rendiment passat.

#### **Inputs**
- `data/fantasy_data.csv`: Dataset en format CSV amb dades de jugadors, equips, i rendiment històric.
- Nom del jugador (input per consola).

#### **Outputs**
- Model entrenat guardat a `model/model.pkl`.
- Predicció de punts per al jugador seleccionat.

---

### 2. `predict_points.py`

#### **Funcionalitat**
Aquest script amplia les funcionalitats de `basic_predict.py` afegint:
- Opció de predir punts per a qualsevol jugador en funció del rival i la localització del partit.
- Mostra una llista de jugadors i equips disponibles per facilitar la selecció.

#### **Inputs**
- `data/fantasy_data.csv`: Dataset en format CSV.
- Nom del jugador, equip rival i localització (input per consola).

#### **Outputs**
- Model entrenat guardat a `model/model.pkl`.
- Predicció dels punts esperats pel jugador seleccionat.

---

### 3. `predict_team.py`

#### **Funcionalitat**
Aquest script prediu el rendiment global d'un equip:
- Prediu els punts per a tots els jugadors d'un equip en un partit concret.
- Mostra els 11 millors jugadors per al partit, segons les prediccions.
- Inclou estadístiques agregades de l'equip.

#### **Inputs**
- `data/fantasy_data.csv`: Dataset en format CSV.
- Nom de l'equip, equip rival i localització (input per consola).

#### **Outputs**
- Llista dels millors 11 jugadors amb punts previstos, valor i estadístiques complementàries.

---

## Requisits

### Llibreries necessàries
- `pandas`
- `numpy`
- `scikit-learn`
- `joblib`



### Format del Dataset (`fantasy_data.csv`)
El dataset ha de contenir com a mínim les següents columnes:
- `player_name`, `team`, `opponent_team`, `was_home`
- `minutes`, `goals_scored`, `assists`, `total_points`, `kickoff_time`

---
