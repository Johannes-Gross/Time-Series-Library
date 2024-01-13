import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from itertools import product
from tqdm import tqdm
import pickle
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Load the dataset
data_path = '../dataset/illness/national_illness.csv'
data = pd.read_csv(data_path)

# Splitting the dataset into training, validation, and test sets
train_ratio = 617 / (617 + 74 + 170)
val_ratio = 74 / (617 + 74 + 170)
train_index = int(len(data) * train_ratio)
val_index = int(len(data) * (train_ratio + val_ratio))
train_data = data.iloc[:train_index]
val_data = data.iloc[train_index:val_index]
test_data = data.iloc[val_index:]

# Combine training and validation data
combined_data = pd.concat([train_data, val_data])

# Normalize the combined dataset
scaler = StandardScaler()
combined_data_scaled = scaler.fit_transform(combined_data.drop(columns=['date']))
combined_data_scaled = pd.DataFrame(combined_data_scaled, columns=combined_data.columns.drop(['date']))

# Normalize the test dataset
test_data_scaled = scaler.transform(test_data.drop(columns=['date']))
test_data_scaled = pd.DataFrame(test_data_scaled, columns=test_data.columns.drop(['date']))

# Define the p, d, and q parameters
p = range(0, 5)
d = range(0, 2)
q = range(0, 5)

# Generate all different combinations of p, d, and q triplets
pdq = [(x[0], x[1], x[2]) for x in list(product(p, d, q))]

# Grid Search for the best ARIMA model based on AIC and BIC
best_aic = np.inf
best_bic = np.inf
best_pdq = None
best_model = None

for param in tqdm(pdq, desc="ARIMA Grid Search"):
    try:
        model = ARIMA(combined_data_scaled['OT'], exog=combined_data_scaled.drop(columns=['OT']), order=param)
        results = model.fit()

        if results.aic < best_aic or results.bic < best_bic:
            best_aic = results.aic
            best_bic = results.bic
            best_pdq = param
            best_model = results
    except:
        continue

print('Best ARIMA{} model - AIC:{} BIC:{}'.format(best_pdq, best_aic, best_bic))

# Print a summary of the best ARIMA model
print("Summary of the Best ARIMA Model:")
print(best_model.summary())

# Forecasting
prediction_length = 24  # Variable for prediction length
forecast = best_model.forecast(steps=prediction_length, exog=test_data_scaled.drop(columns=['OT']).iloc[:prediction_length])

# Calculate MSE and MAE for the forecast
mse = mean_squared_error(test_data_scaled['OT'][:prediction_length], forecast)
mae = mean_absolute_error(test_data_scaled['OT'][:prediction_length], forecast)

# Plot the forecast against the test_data
plt.figure(figsize=(12, 6))
plt.plot(range(prediction_length), test_data_scaled['OT'][:prediction_length], label='Actual Scaled')
plt.plot(range(prediction_length), forecast, label='Forecast Scaled', color='red')
plt.title(f'ARIMA Forecast vs Actual for Next {prediction_length} Steps (Scaled)')
plt.xlabel('Time Steps')
plt.ylabel('Scaled OT')
plt.legend()
plt.grid(True)
plt.savefig(f'../test_results/ARIMA/Illness/ILI_forecast_next_{prediction_length}_steps_scaled.png')
plt.show()

print(f"Scaled Forecasting Results for {prediction_length} Steps - MSE: {mse}, MAE: {mae}")

# Save results to a file
results_dict = {'scaled_mse': mse, 'scaled_mae': mae, 'prediction_length': prediction_length}
with open('../test_results/ARIMA/Illness/ARIMA_Illness_Scaled_Forecast_Results.pkl', 'wb') as file:
    pickle.dump(results_dict, file)
