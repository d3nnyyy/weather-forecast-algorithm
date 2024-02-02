import logging
import os

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_openmeteo_client():
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


def make_api_request(openmeteo, latitude, longitude, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", "relative_humidity_2m", "apparent_temperature", "precipitation", "rain",
                   "surface_pressure", "cloud_cover", "wind_speed_100m", "wind_direction_100m"]
    }
    return openmeteo.weather_api(url, params=params)


def process_api_response(response):
    hourly = response.Hourly()
    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s"),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
        "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy(),
        "apparent_temperature": hourly.Variables(2).ValuesAsNumpy(),
        "precipitation": hourly.Variables(3).ValuesAsNumpy(),
        "rain": hourly.Variables(4).ValuesAsNumpy(),
        "surface_pressure": hourly.Variables(5).ValuesAsNumpy(),
        "cloud_cover": hourly.Variables(6).ValuesAsNumpy(),
        "wind_speed_100m": hourly.Variables(7).ValuesAsNumpy(),
        "wind_direction_100m": hourly.Variables(8).ValuesAsNumpy()
    }
    return pd.DataFrame(data=hourly_data)


def save_to_csv(dataframe, csv_filename):
    dataframe.to_csv(csv_filename, index=False)
    logger.info(f"Data saved to {csv_filename}")


def process_city_data(openmeteo, city_row, start_date, end_date):
    latitude = city_row['lat']
    longitude = city_row['lon']
    city_name = city_row['city'].lower()

    csv_filename = f'data/{city_name}_weather.csv'

    if os.path.exists(csv_filename):
        logger.info(f"Weather data for {city_name} already exists. Skipping...")
        return

    try:
        responses = make_api_request(openmeteo, latitude, longitude, start_date, end_date)
        response = responses[0]
        dataframe = process_api_response(response)
        save_to_csv(dataframe, csv_filename)
        logger.info(f"Weather data for {city_name} processed and saved to {csv_filename}")
    except Exception as e:
        logger.error(f"Error processing data for {city_name}: {str(e)}")


if __name__ == '__main__':
    openmeteo = setup_openmeteo_client()
    cities_df = pd.read_csv('data/list_of_cities.csv')

    start_date = pd.to_datetime('today') - pd.Timedelta(days=365 * 5)
    end_date = pd.to_datetime('today') - pd.Timedelta(days=7)

    for index, city_row in cities_df.iterrows():
        process_city_data(openmeteo, city_row, start_date, end_date)