# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 23:36:35 2024

@author: nildi
"""
#1 IDEES PER A MILLORAR EL CODI PROVA2: Optimitzar hiperparametres
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best Hyperparameters: {grid_search.best_params_}")
rf = grid_search.best_estimator_

#2 IDDEES PER A MILLORAR EL CODI PROVA2 Afegir visualitzacions
# Distribució dels errors: Pots visualitzar com es distribueixen els errors entre les prediccions i els valors reals:
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Punts reals')
plt.ylabel('Punts predits')
plt.title('Comparació entre punts reals i predits')
plt.show()

# Importància de les característiques: Analitzar quines característiques tenen més impacte en les prediccions:
importances = rf.feature_importances_
sorted_indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[sorted_indices])
plt.xticks(range(len(importances)), X_df.columns[sorted_indices], rotation=90)
plt.title('Importància de les característiques')
plt.show()

#3 IDEES PER A MILLORAR EL CODI PROVA2
# Com que estàs treballant amb sèries temporals, pots utilitzar una validació temporal amb TimeSeriesSplit en lloc de dividir les dades aleatòriament:
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Fold MAE: {mae}")


#4 IDEES PER A MILLORAR EL CODI PROVA2
# A més del Mean Absolute Error (MAE), pots utilitzar altres mètriques per avaluar el model:
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")

#5 IDEES PER A MILLORAR EL CODI PROVA2: Generar csv per a veure les prediccions
results = pd.DataFrame({'Real Points': y_test, 'Predicted Points': y_pred})
results.to_csv('predictions.csv', index=False)
print("Resultats exportats a predictions.csv")


#6 IDEES PER A MILLORAR EL CODI :
# Pots provar altres models com Gradient Boosting (e.g., XGBoost, LightGBM) o fins i tot models de regressió lineal:
from sklearn.ensemble import GradientBoostingRegressor

gbm = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gbm.fit(X_train, y_train)
y_pred_gbm = gbm.predict(X_test)

print(f"MAE per GBM: {mean_absolute_error(y_test, y_pred_gbm)}")
