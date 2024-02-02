import os

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

# Set up the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Read the list of cities from CSV
cities_df = pd.read_csv('data/list_of_cities.csv')

# Iterate through each city in the CSV file
for index, city_row in cities_df.iterrows():
    latitude = city_row['lat']
    longitude = city_row['lon']
    city_name = city_row['city'].lower()

    # Check if the CSV file already exists for the city
    csv_filename = f'data/{city_name}_weather.csv'
    if os.path.exists(csv_filename):
        print(f"Weather data for {city_name} already exists. Skipping...")
        continue

    # Make API request for weather data
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": "2022-01-01",
        "end_date": "2024-01-30",
        "hourly": ["temperature_2m", "relative_humidity_2m", "apparent_temperature", "precipitation", "rain",
                   "surface_pressure", "cloud_cover", "wind_speed_100m", "wind_direction_100m"]
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
    hourly_apparent_temperature = hourly.Variables(2).ValuesAsNumpy()
    hourly_precipitation = hourly.Variables(3).ValuesAsNumpy()
    hourly_rain = hourly.Variables(4).ValuesAsNumpy()
    hourly_surface_pressure = hourly.Variables(5).ValuesAsNumpy()
    hourly_cloud_cover = hourly.Variables(6).ValuesAsNumpy()
    hourly_wind_speed_100m = hourly.Variables(7).ValuesAsNumpy()
    hourly_wind_direction_100m = hourly.Variables(8).ValuesAsNumpy()

    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s"),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "temperature_2m": hourly_temperature_2m,
        "relative_humidity_2m": hourly_relative_humidity_2m,
        "apparent_temperature": hourly_apparent_temperature,
        "precipitation": hourly_precipitation,
        "rain": hourly_rain,
        "surface_pressure": hourly_surface_pressure,
        "cloud_cover": hourly_cloud_cover,
        "wind_speed_100m": hourly_wind_speed_100m,
        "wind_direction_100m": hourly_wind_direction_100m
    }

    # Save DataFrame to CSV file with city name
    hourly_dataframe = pd.DataFrame(data=hourly_data)
    hourly_dataframe.to_csv(csv_filename, index=False)

    print(f"Weather data for {city_name} saved to {csv_filename}")
