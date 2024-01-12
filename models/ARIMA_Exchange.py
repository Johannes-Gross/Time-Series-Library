# Import necessary libraries
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from itertools import product
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings("ignore")

# Load the dataset
data_path = '../dataset/exchange_rate/exchange_rate.csv'
data = pd.read_csv(data_path)

# Splitting the dataset into training, validation, and test sets
train_ratio = 5120 / (5120 + 665 + 1422)
val_ratio = 665 / (5120 + 665 + 1422)
train_index = int(len(data) * train_ratio)
val_index = int(len(data) * (train_ratio + val_ratio))
train_data = data.iloc[:train_index]
val_data = data.iloc[train_index:val_index]
test_data = data.iloc[val_index:]

# Define the p, d, and q parameters
p = range(0, 5)
d = range(0, 2)
q = range(0, 5)

# Generate all different combinations of p, d, and q triplets
pdq = [(x[0], x[1], x[2]) for x in list(product(p, d, q))]

# Grid Search based on MSE and MAE on validation set
best_mse = np.inf
best_mae = np.inf
best_pdq = None
best_model = None

for param in tqdm(pdq, desc="ARIMA Grid Search"):
    model = ARIMA(train_data['OT'], order=param, exog=train_data.drop(columns=['date', 'OT']))  # ARIMA with exogenous variables
    results = model.fit()
    forecast = results.forecast(steps=len(val_data), exog=val_data.drop(columns=['date', 'OT']))  # Forecast with exogenous variables

    mse = mean_squared_error(val_data['OT'], forecast)
    mae = mean_absolute_error(val_data['OT'], forecast)

    if mse < best_mse and mae < best_mae:
        best_mse = mse
        best_mae = mae
        best_pdq = param
        best_model = results

print('Best ARIMA{} model - MSE:{} MAE:{}'.format(best_pdq, best_mse, best_mae))

# Print a summary of the best ARIMA model
print("Summary of the Best ARIMA Model:")
print(best_model.summary())

# Define prediction lengths
pred_lengths = [96, 192, 336, 720]

# Forecasting for different prediction lengths on the test set
results_dict = {}
total_mse, total_mae = 0, 0

for pred_len in pred_lengths:
    exog_test = test_data.drop(columns=['date', 'OT']).iloc[:pred_len]  # Exogenous data for each prediction length
    forecast = best_model.forecast(steps=pred_len, exog=exog_test)  # Forecast with exogenous variables on test set

    mse = mean_squared_error(test_data['OT'][:pred_len], forecast)
    mae = mean_absolute_error(test_data['OT'][:pred_len], forecast)

    results_dict[pred_len] = {'mse': mse, 'mae': mae}
    total_mse += mse
    total_mae += mae

# Display results for each prediction length
for pred_len, result in results_dict.items():
    print(f"Results for pred_len={pred_len}:")
    print(f"MSE: {result['mse']}, MAE: {result['mae']}\n")

# Calculate and add the average MSE and MAE to the results_dict
avg_mse = total_mse / len(pred_lengths)
avg_mae = total_mae / len(pred_lengths)
results_dict['average'] = {'mse': avg_mse, 'mae': avg_mae}

# Print average results
print(f"Average Results:")
print(f"Average MSE: {results_dict['average']['mse']}, Average MAE: {results_dict['average']['mae']}")

# Save results_dict to a file using pickle
with open('../results/ARIMA/ARIMA_Exchange.pkl', 'wb') as file:
    pickle.dump(results_dict, file)
