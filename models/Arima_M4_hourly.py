import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from itertools import product
from tqdm import tqdm
import pickle
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Function definitions for the new metrics (SMAPE, MAPE, MASE)
def smape(actual, forecast):
    return 100 * np.mean(2 * np.abs(forecast - actual) / (np.abs(actual) + np.abs(forecast)))

def mape(actual, forecast):
    return 100 * np.mean(np.abs((actual - forecast) / actual))

def mase(training_series, testing_series, prediction_series):
    n = len(training_series)
    d = np.mean(np.abs(np.diff(training_series)))
    errors = np.abs(testing_series - prediction_series)
    return np.mean(errors) / d

# Naïve2 forecasting function
def naive2_forecast(series, prediction_length=48):
    return series[-prediction_length:]

# Script modification for the M4 dataset
def process_m4_series(train, test, prediction_length=48):
    # Normalize the training dataset
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1)).flatten()

    # Normalize the test dataset
    test_scaled = scaler.transform(test.values.reshape(-1, 1)).flatten()

    # Define the p, d, and q parameters
    p = range(0, 5)
    d = range(0, 2)
    q = range(0, 5)
    pdq = [(x[0], x[1], x[2]) for x in list(product(p, d, q))]

    # Grid Search for the best ARIMA model
    best_aic = np.inf
    best_pdq = None
    best_model = None

    for param in tqdm(pdq, desc="ARIMA Grid Search"):
        try:
            model = ARIMA(train_scaled, order=param)
            results = model.fit()
            if results.aic < best_aic:
                best_aic = results.aic
                best_pdq = param
                best_model = results
        except:
            continue

    # Forecasting
    forecast_scaled = best_model.forecast(steps=prediction_length)

    # Calculate the new metrics on scaled values
    smape_value = smape(test_scaled[:prediction_length], forecast_scaled)
    mape_value = mape(test_scaled[:prediction_length], forecast_scaled)
    mase_value = mase(train_scaled, test_scaled[:prediction_length], forecast_scaled)

     # Calculate the metrics for the forecast
    smape_forecast = smape(test_scaled[:prediction_length], forecast_scaled)
    mase_forecast = mase(train_scaled, test_scaled[:prediction_length], forecast_scaled)

    # Calculate the metrics for the Naïve2 model
    naive2_prediction = naive2_forecast(test_scaled, prediction_length)
    smape_naive2 = smape(test_scaled[:prediction_length], naive2_prediction)
    mase_naive2 = mase(train_scaled, test_scaled[:prediction_length], naive2_prediction)

    # Calculate OWA
    owa = 0.5 * (smape_forecast / smape_naive2 + mase_forecast / mase_naive2)

    # Plotting
    plt.figure(figsize=(12, 6))
    #plt.plot(train_scaled, label='Training Data')
    plt.plot(range(len(train_scaled), len(train_scaled) + len(test_scaled)), test_scaled, label='Test Data')
    plt.plot(range(len(train_scaled), len(train_scaled) + prediction_length), forecast_scaled, label='Forecast', color='red')
    plt.title('ARIMA Forecast vs Actual (Scaled)')
    plt.xlabel('Time Steps')
    plt.ylabel('Scaled Values')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'../test_results/ARIMA/M4/Hourly/Hourly_forecast_next_{prediction_length}_steps_scaled.png')
    plt.show()

    return {'smape': smape_value, 'mape': mape_value, 'mase': mase_value, 'owa': owa,'best_model_summary': best_model.summary()}

# Example usage
# Load the M4 dataset
train_data_path = '../dataset/m4/Hourly-train.csv' 
test_data_path = '../dataset/m4/Hourly-test.csv'  # Replace with your file path
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# Process the first series in the dataset
results = process_m4_series(train_data.iloc[0, 1:].dropna(), test_data.iloc[0, 1:].dropna())
print(results['best_model_summary'])
print("SMAPE:", results['smape'])
print("MAPE:", results['mape'])
print("MASE:", results['mase'])
print("OWA:", results['owa'])

# Save results to a file
with open('../test_results/ARIMA/M4/Hourly/ARIMA_M4_Scaled_Forecast_Results.pkl', 'wb') as file:
    pickle.dump(results, file)
