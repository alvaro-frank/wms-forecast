"""
Visualizes Prophet forecasts for a specific product hierarchy and brand.
- Filters test_data for selected BRAND and PRODUCTHIERARCHY3
- Trains Prophet, forecasts, plots results, and prints RMSE/MAE
"""

import pandas as pd
import matplotlib.pyplot as plt

from prophet import Prophet

from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils.data_split import test_data

# Filter test data for the specified brand and product hierarchy
filtered_df = test_data[
    (test_data['BRAND'] == brand_code) &
    (test_data['PRODUCTHIERARCHY3'] == hier_code)
].copy()

# Get product and brand descriptions for plot titles
match = filtered_df.loc[filtered_df['PRODUCTHIERARCHY3'] == hier_code, 'PRODUCTHIERARCHY3NAME']
description = match.iloc[0] if not match.empty else "Unknown Product"

match = filtered_df.loc[filtered_df['BRAND'] == brand_code, 'BRANDNAME']
brand_description = match.iloc[0] if not match.empty else "Unknown Brand"

# Prepare time series for Prophet (ds = date, y = quantity)
ts = (
    filtered_df
    .groupby("DATE")["QUANTITY"]
    .sum()
    .reset_index()
    .rename(columns={"DATE": "ds", "QUANTITY": "y"})
)

# Confirm structure
print(ts.head())
print(ts.columns)

# Train/test split for Prophet
train_size = int(len(ts) * 0.7)
train_df = ts[:train_size]
test_df = ts[train_size:]

# Train Prophet model
model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
model.fit(train_df)

# Forecast for the test period
future = model.make_future_dataframe(periods=len(test_df))
forecast = model.predict(future)

# Plot forecast and actuals
fig = model.plot(forecast)
plt.title(f'Prophet Forecast for PRODUCTHIERARCHY3 = {hier_code} ({description}) and BRAND = {brand_code} ({brand_description})')
plt.xlabel('Date')
plt.ylabel('Quantity')
plt.grid(True)
plt.legend()

# Merge actuals and predictions for metric calculation
merged = test_df.merge(forecast[['ds', 'yhat']], on='ds', how='left')

# Drop any rows without predictions
merged = merged.dropna(subset=["yhat"])

# Compute and print RMSE and MAE
rmse = mean_squared_error(merged['y'], merged['yhat'], squared=False)
mae = mean_absolute_error(merged['y'], merged['yhat'])
print(f"Prophet RMSE: {rmse:.2f}")
print(f"Prophet MAE: {mae:.2f}")