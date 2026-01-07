"""
Evaluates LSTM model performance on original (unnormalized) scale.
- Uses trained LSTM, windowed data, and normalization stats
- Prints MAE for validation and test sets on both normalized and original scales
"""

from sklearn.metrics import mean_absolute_error

from models.lstm import lstm, wide_window, train_mean, train_std, val_performance, performance

def evaluate_on_original_scale(model, window, train_mean, train_std, quantity_column='QUANTITY'):
    """
    Computes MAE on the original scale by reversing normalization.
    """
    actual_values = []
    predicted_values = []
    for inputs, labels in window:
        predictions = model.predict(inputs)

        # Reverse normalization for predictions
        predicted_quantity_original = (predictions.squeeze() * train_std[quantity_column]) + train_mean[quantity_column]
        predicted_values.extend(predicted_quantity_original.flatten())

        # Reverse normalization for actual values
        actual_quantity_original = (labels.numpy().squeeze() * train_std[quantity_column]) + train_mean[quantity_column]
        actual_values.extend(actual_quantity_original.flatten())

    # Calculate MAE on original scale
    mae_original_scale = mean_absolute_error(actual_values, predicted_values)
    return mae_original_scale

# Store MAE results for validation and test sets
val_mae_original = {}
test_mae_original = {}

val_mae_original['LSTM'] = evaluate_on_original_scale(lstm, wide_window.val, train_mean, train_std)
test_mae_original['LSTM'] = evaluate_on_original_scale(lstm, wide_window.test, train_mean, train_std)

# Print MAE for normalized scale (validation)
print('Validation MAE (Normalized Scale):')
for name, value in val_performance.items():
  print(f'{name:12s}: {value[1]:0.4f}')

# Print MAE for normalized scale (test)
print('\nTest MAE (Normalized Scale):')
for name, value in performance.items():
  print(f'{name:12s}: {value[1]:0.4f}')