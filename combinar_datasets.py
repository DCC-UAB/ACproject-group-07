# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:49:57 2024

@author: nildi
"""

import pandas as pd

# Carregar els datasets
df1 = pd.read_csv('model_evaluation_results.csv')
df2 = pd.read_csv('model_evaluation_results_5.csv')
df3 = pd.read_csv('model_evaluation_results_10.csv')
df4 = pd.read_csv('model_evaluation_results_15.csv')

# Combinar els datasets per files
combined_df = pd.concat([df1, df2, df3, df4], ignore_index=True)

# Mostrar les primeres files
print(combined_df.head())

# Guardar el dataset combinat
combined_df.to_csv('combined_dataset.csv', index=False)