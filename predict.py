"""
This Python script loads previously trained Prophet time series forecasting models
for various weather variables, defines a function predict_weather
that generates forecasts for a specified start date and number of periods,
and returns the combined results in a DataFrame.
"""

import os
import pickle

import pandas as pd

models_folder = 'models'

variables = ['temp_2', 'hum_2', 'temp_a', 'precip', 'rain', 'press', 'cloud', 'w_speed', 'w_dir']

loaded_models = {}
for variable in variables:
    file_path = os.path.join(models_folder, f'{variable}_prophet_model.pkl')
    with open(file_path, 'rb') as f:
        loaded_models[variable] = pickle.load(f)


def predict_weather(start_date, periods):
    result_df = pd.DataFrame({'ds': pd.date_range(start=start_date, periods=periods, freq='h')})
    combined_df = pd.DataFrame({'ds': result_df['ds']})
    for variable in variables:
        result = loaded_models[variable].predict(result_df)
        combined_df[variable] = result['yhat']
    return combined_df
