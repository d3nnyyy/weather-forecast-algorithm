import os
import pickle
from math import radians, sin, cos, sqrt, atan2

import pandas as pd


def is_big_city(latitude, longitude, threshold=0.01):
    cities_df = pd.read_csv('data/list_of_cities.csv')
    for index, row in cities_df.iterrows():

        lat = row['lat']
        lon = row['lon']

        if abs(latitude - lat) < threshold and abs(longitude - lon) < threshold:
            return True, row['city']
    return False, None


def get_models_for_big_city(city_name):
    models_folder = 'models'
    city_folder = os.path.join(models_folder, str(city_name))
    models = {}

    if not os.path.exists(city_folder):
        return models

    for variable in os.listdir(city_folder):
        if variable.endswith('.pkl'):
            model_path = os.path.join(city_folder, variable)
            with open(model_path, 'rb') as f:
                models[variable] = pickle.load(f)

    return models


def predict_weather_for_big_city(start_date, periods, city_name):
    result_df = pd.DataFrame({'ds': pd.date_range(start=start_date, periods=periods, freq='h')})
    combined_df = pd.DataFrame({'ds': result_df['ds']})

    loaded_models = get_models_for_big_city(city_name)
    for variable in loaded_models:
        result = loaded_models[variable].predict(result_df)
        combined_df[variable] = result['yhat']
        combined_df.columns = combined_df.columns.str.replace('.pkl', '')
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

    # Initialize a list to store distances, corresponding city data, and their distances
    city_distances = []

    for index, city in cities_df.iterrows():
        # Calculate haversine distance for each city
        distance = haversine(latitude, longitude, city['lat'], city['lon'])
        city_distances.append((index, distance))

    # Sort the distances and get the indices of the num_cities nearest cities
    nearest_indices = sorted(city_distances, key=lambda x: x[1])[:num_cities]

    # Get the city data and distances for the nearest cities
    nearest_cities_data = [(cities_df.iloc[index], distance) for index, distance in nearest_indices]

    return nearest_cities_data


def calculate_weighted_weather(start_date, periods, nearest_cities):
    inverted_distances = [1 / distance for _, distance in nearest_cities]
    total_inverted_distance = sum(inverted_distances)
    distances_weights = [distance / total_inverted_distance for distance in inverted_distances]

    weather_dfs = [predict_weather_for_big_city(start_date, periods, city['city']) for city, _ in nearest_cities]

    # return pd.concat(weather_dfs).groupby(level=0).apply(lambda x: (x * distances_weights).sum()).reset_index()

    #return mean for now
    return pd.concat(weather_dfs).groupby(level=0).mean().reset_index()


def predict_weather_for_small_city(start_date, periods, latitude, longitude):
    nearest_cities_data = find_nearest_big_cities(latitude, longitude)
    weighted_weather = calculate_weighted_weather(start_date, periods, nearest_cities_data)
    return weighted_weather


def predict_weather(start_date, periods, latitude, longitude):
    big_city, city_name = is_big_city(latitude, longitude)

    if big_city:
        return predict_weather_for_big_city(start_date, periods, city_name)
    else:
        return predict_weather_for_small_city(start_date, periods, latitude, longitude)


print(predict_weather("2023-07-01", 5, 49.5557716, 25.591886)['temp_2'])
print(predict_weather("2023-07-01", 5, 50.130550, 25.259340)['temp_2'])
