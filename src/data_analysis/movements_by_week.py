"""
Average products moved per weekday (by year).
- Uses movimentos_saida_mercadoria (YEAR, WEEKDAY, QUANTITY)
- Plots average quantity per weekday for each year
"""

import pandas as pd
import matplotlib.pyplot as plt

from utils.data_split import movimentos_saida_mercadoria

# Group data by year and weekday, then calculate the average quantity moved
sales_by_weekday = movimentos_saida_mercadoria.groupby(['YEAR', 'WEEKDAY'])['QUANTITY'].mean().reset_index()

# Get all unique years present in the data
unique_years = sales_by_weekday['YEAR'].unique()

# Define labels for weekdays (0=Mon, 6=Sun)
weekday_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

# Set up the plot size
plt.figure(figsize=(12, 6))

# Plot a line for each year showing average products moved per weekday
for year in unique_years:
    data_for_year = sales_by_weekday[sales_by_weekday['YEAR'] == year]
    plt.plot(
        data_for_year['WEEKDAY'],
        data_for_year['QUANTITY'],
        marker='o',
        label=str(year),
        linestyle='-',
        markersize=8
    )

# Add plot title and axis labels
plt.title('Average Products Moved per Weekday by Year')
plt.xlabel('Weekday')
plt.ylabel('Average Products Moved')

# Set x-axis ticks to weekday names
plt.xticks(range(7), weekday_labels)

# Show legend with year labels
plt.legend(title='Year')

# Adjust layout and display the plot
plt.tight_layout()
plt.show()