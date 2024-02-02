import os
import pickle

import pandas as pd
from prophet import Prophet


def load_city_data(city_name):
    file_path = f"data/{city_name}_weather.csv"
    df = pd.read_csv(file_path)

    column_mapping = {
        'date': 'ds',
        'temperature_2m': 'y_temp_2',
        'relative_humidity_2m': 'y_hum_2',
        'apparent_temperature': 'y_temp_a',
        'precipitation': 'y_precip',
        'rain': 'y_rain',
        'surface_pressure': 'y_press',
        'cloud_cover': 'y_cloud',
        'wind_speed_100m': 'y_w_speed',
        'wind_direction_100m': 'y_w_dir'
    }
    df = df.rename(columns=column_mapping)

    return df


def load_hyperparameters(model_name):
    hyperparameters_df = pd.read_csv("data/models_hyperparameters.csv")
    model_params = hyperparameters_df[hyperparameters_df['model'] == model_name].iloc[0]
    return model_params['changepoint_prior_scale'], model_params['seasonality_prior_scale']


models_directory = "models"
if not os.path.exists(models_directory):
    os.makedirs(models_directory)
    print(f"Directory {models_directory} created")
else:
    print(f"Directory {models_directory} already exists")

models = {}

cities_df = pd.read_csv("data/list_of_cities.csv")
cities = cities_df['city'].str.lower()

for city_name in cities:
    # Check if models for the current city already exist
    model_directory_city = os.path.join(models_directory, city_name)

    # Load data for the current city
    df = load_city_data(city_name)

    # Initialize the dictionary for the current city
    city_models = {}

    # Iterate through each variable
    variables = ['temp_2', 'hum_2', 'temp_a', 'precip', 'rain', 'press', 'cloud', 'w_speed', 'w_dir']
    for variable in variables:

        # Prepare data
        column_name = f'y_{variable}'
        df_temp = df[['ds', column_name]].rename(columns={column_name: 'y'})

        # Feature engineering: Adding lag variables
        for i in range(1, 6):
            df_temp[f'y_{variable}_lag_{i}'] = df_temp['y'].shift(i)

        # Drop rows with NaN values after adding lag variables
        df_temp = df_temp.dropna()

        # Load hyperparameters for the current model
        changepoint_prior_scale, seasonality_prior_scale = load_hyperparameters(variable)

        # Check if the model file exists in the city folder
        model_filename = os.path.join(model_directory_city, f"{variable}.pkl")
        if os.path.exists(model_filename):
            print(f"Model for {variable} in {city_name} already exists.")
        else:
            # Create and train the model with hyperparameters
            model = Prophet(changepoint_prior_scale=changepoint_prior_scale,
                            seasonality_prior_scale=seasonality_prior_scale)
            model.fit(df_temp)
            city_models[variable] = model

            # Save the model to the city folder
            with open(model_filename, 'wb') as f:
                pickle.dump(model, f)
            print(f"Model for {variable} in {city_name} trained and saved to {model_filename}")

    models[city_name] = city_models