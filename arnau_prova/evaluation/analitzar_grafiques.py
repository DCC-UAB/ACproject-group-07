import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Llegeix les dades globals des de la variable o fitxer
file_path = 'XGBoost_results.csv'

df = pd.read_csv(file_path)

plt.style.use('ggplot')

# 1. Gràfica: Comparació del MAE de validació i test per `n_prev_games`, separant per lr
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='n_prev_games', y='validation_MAE', hue='learning_rate', marker='o')
sns.lineplot(data=df, x='n_prev_games', y='test_MAE', hue='learning_rate', marker='o', linestyle='--')
plt.title('Validation MAE vs Test MAE per n_prev_games (amb learning_rate)')
plt.xlabel('n_prev_games')
plt.ylabel('MAE')
plt.legend(title='Learning Rate')
plt.grid(True)
plt.show()

# 2. Gràfica: RMSE de validació i test per `n_prev_games`, separant per lr
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='n_prev_games', y='validation_RMSE', hue='learning_rate', marker='o')
sns.lineplot(data=df, x='n_prev_games', y='test_RMSE', hue='learning_rate', marker='o', linestyle='--')
plt.title('Validation RMSE vs Test RMSE per n_prev_games (amb learning_rate)')
plt.xlabel('n_prev_games')
plt.ylabel('RMSE')
plt.legend(title='Learning Rate')
plt.grid(True)
plt.show()

# 3. Gràfica: Impacte del learning_rate en Validation MAE
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='learning_rate', y='validation_MAE')
plt.title('Impacte del Learning Rate en Validation MAE')
plt.xlabel('Learning Rate')
plt.ylabel('Validation MAE')
plt.grid(True)
plt.show()

# 4. Gràfica: Iteracions òptimes (`best_iteration`) per `n_prev_games` i `learning_rate`
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='n_prev_games', y='best_iteration', hue='learning_rate', style='learning_rate', s=100)
plt.title('Best Iteration per n_prev_games i Learning Rate')
plt.xlabel('n_prev_games')
plt.ylabel('Best Iteration')
plt.legend(title='Learning Rate')
plt.grid(True)
plt.show()

# 5. Gràfica: Comparació de MAE i RMSE segons els paràmetres
g = sns.FacetGrid(df, col="learning_rate", hue="n_estimators", height=5, aspect=1.5)
g.map(sns.lineplot, "n_prev_games", "test_MAE", marker="o")
g.set_axis_labels("n_prev_games", "Test MAE")
g.set_titles("Learning Rate: {col_name}")
g.add_legend()
plt.show()

# 6. Gràfica: Comparació de MAE en funció del nombre d'estimadors
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='n_estimators', y='test_MAE')
plt.title('Impacte del Nombre d\'Estimadors en Test MAE')
plt.xlabel('n_estimators')
plt.ylabel('Test MAE')
plt.grid(True)
plt.show()
