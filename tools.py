import logging
import os
import pickle
from math import radians, sin, cos, sqrt, atan2

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NO_NEARBY_CITIES_MESSAGE = "No nearby big cities found for weather prediction."
LOCATION_TOO_FAR_MESSAGE = "Weather prediction not possible as location is too far from any regional center."
CITY_NOT_FOUND_MESSAGE = "City not found in the models directory."
WEATHER_NOT_POSSIBLE_MESSAGE = "Weather prediction not possible due to lack of nearby big cities."


def get_absolute_path(relative_path):
    current_script_directory = os.path.dirname(__file__)
    absolute_path = os.path.join(current_script_directory, relative_path)
    return absolute_path


def is_big_city(latitude, longitude, threshold=0.01):
    csv_file_path = get_absolute_path('data/list_of_cities.csv')
    cities_df = pd.read_csv(csv_file_path)
    for index, row in cities_df.iterrows():

        lat = row['lat']
        lon = row['lon']

        if abs(latitude - lat) < threshold and abs(longitude - lon) < threshold:
            return True, row['city']
    return False, None


def get_models_for_big_city(city_name):
    models_folder = 'models'
    city_folder = os.path.join(models_folder, str(city_name.lower()))
    models = {}

    if not os.path.exists(city_folder):
        logger.error(f"{CITY_NOT_FOUND_MESSAGE} City {city_name}")
        return models

    for variable in os.listdir(city_folder):
        if variable.endswith('.pkl'):
            model_path = os.path.join(city_folder, variable)
            variable_name = variable.replace('.pkl', '')
            with open(model_path, 'rb') as f:
                models[variable_name] = pickle.load(f)
    return models


def predict_weather_for_big_city(start_date, periods, city_name):
    result_df = pd.DataFrame({'ds': pd.date_range(start=start_date, periods=periods, freq='h')})
    combined_df = pd.DataFrame({'ds': result_df['ds']})

    loaded_models = get_models_for_big_city(city_name)

    if not loaded_models:
        return combined_df

    for variable in loaded_models:
        result = loaded_models[variable].predict(result_df)

        combined_df[variable] = result['yhat']

    return combined_df


def haversine(lat1, lon1, lat2, lon2):
    EARTH_RADIUS_KM = 6371

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = EARTH_RADIUS_KM * c
    return distance


def find_nearest_big_cities(latitude, longitude, num_cities=3):
    cities_df = pd.read_csv('data/list_of_cities.csv')

    city_distances = []

    for index, city in cities_df.iterrows():
        distance = haversine(latitude, longitude, city['lat'], city['lon'])
        city_distances.append((index, distance))

    nearest_indices = sorted(city_distances, key=lambda x: x[1])[:num_cities]

    nearest_cities_data = [(cities_df.iloc[index], distance) for index, distance in nearest_indices]

    return nearest_cities_data


def calculate_weighted_weather(start_date, periods, nearest_cities):
    if not nearest_cities:
        return pd.DataFrame({'ds': pd.date_range(start=start_date, periods=periods, freq='h')})

    inverted_distances = [1 / distance for _, distance in nearest_cities]
    total_inverted_distance = sum(inverted_distances)
    distances_weights = [distance / total_inverted_distance for distance in inverted_distances]

    weather_dfs = [predict_weather_for_big_city(start_date, periods, city['city']) for city, _ in nearest_cities]

    # return pd.concat(weather_dfs).groupby(level=0).apply(lambda x: (x * distances_weights).sum()).reset_index()

    # return mean for now
    return pd.concat(weather_dfs).groupby(level=0).mean().reset_index()


def predict_weather_for_small_city(start_date, periods, latitude, longitude, max_distance_for_prediction=200):
    nearest_cities_data = find_nearest_big_cities(latitude, longitude)

    if not nearest_cities_data:
        logger.error(NO_NEARBY_CITIES_MESSAGE)
        return pd.DataFrame({'ds': pd.date_range(start=start_date, periods=periods, freq='h'),
                             'message': WEATHER_NOT_POSSIBLE_MESSAGE})

    nearest_distance = nearest_cities_data[0][1]

    if nearest_distance > max_distance_for_prediction:
        logger.error(f"{LOCATION_TOO_FAR_MESSAGE} (Distance: {nearest_distance} km)")
        return pd.DataFrame({'ds': pd.date_range(start=start_date, periods=periods, freq='h'),
                             'message': LOCATION_TOO_FAR_MESSAGE})

    weighted_weather = calculate_weighted_weather(start_date, periods, nearest_cities_data)
    return weighted_weather


def predict_weather(start_date, periods, latitude, longitude):
    big_city, city_name = is_big_city(latitude, longitude)

    if big_city:
        return predict_weather_for_big_city(start_date, periods, city_name)
    else:
        return predict_weather_for_small_city(start_date, periods, latitude, longitude)
