"""
Detects and visualizes outliers in daily product movement quantities.
- Uses movimentos_saida_mercadoria_grouped_date (QUANTITY)
- Prints number of outliers and shows a boxplot of QUANTITY
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

from utils.data_split import movimentos_saida_mercadoria_grouped_date

# Calculate z-scores for the QUANTITY column to identify outliers
z_scores = np.abs(stats.zscore(movimentos_saida_mercadoria_grouped_date["QUANTITY"]))
outliers = movimentos_saida_mercadoria_grouped_date[z_scores > 3]

# Print the number of detected outliers
print(f"Number of outliers: {len(outliers)}\n")
outliers.head()

# Plot a boxplot to visualize the distribution and outliers in QUANTITY
plt.figure(figsize=(12, 6))
sns.boxplot(x=movimentos_saida_mercadoria_grouped_date["QUANTITY"])
plt.title("Boxplot of QUANTITY")
plt.show()