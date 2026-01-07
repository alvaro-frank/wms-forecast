"""
Monthly average products moved (per year).
- Uses movimentos_saida_mercadoria (YEAR, MONTH, QUANTITY)
- Prints monthly means and plots a line per year
"""

import pandas as pd
import matplotlib.pyplot as plt

from utils.data_split import movimentos_saida_mercadoria

# Group data by year and month, then calculate the average quantity moved
sales_by_month = movimentos_saida_mercadoria.groupby(['YEAR', 'MONTH'])['QUANTITY'].mean().reset_index()

# Print the resulting DataFrame for inspection
print(sales_by_month)
print('\n------------------------------------\n')

# Get all unique years present in the data
unique_years = sales_by_month['YEAR'].unique()

# Set up the plot size
plt.figure(figsize=(12, 6))

# Plot a line for each year showing average products moved per month
for year in unique_years:
    data_for_year = sales_by_month[sales_by_month['YEAR'] == year]
    plt.plot(
        data_for_year['MONTH'],
        data_for_year['QUANTITY'],
        marker='o',
        label=str(year),
        linestyle='-',
        markersize=8
    )

# Add plot title and axis labels
plt.title('Monthly Average Products Moved for Sales per Year')
plt.xlabel('Month')
plt.ylabel('Average Products Moved')

# Set x-axis ticks to month names
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

# Show legend with year labels
plt.legend(title='Year')

# Adjust layout and display the plot
plt.tight_layout()
plt.show()