"""
This Flask application defines a /predict endpoint that takes
start_date and optional periods parameters from a GET request,
uses the predict_weather function to generate weather forecasts,
converts the results to JSON, and returns the predictions when the script is run.
"""

import json

from flask import Flask, request
from flask_cors import CORS

from tools import predict_weather, calculate_daily_medians

app = Flask(__name__)
CORS(app)


@app.route('/predict', methods=['GET'])
def predict():
    start_date = request.args.get('start_date')
    periods = int(request.args.get('periods', 24))
    latitude = float(request.args.get('latitude'))
    longitude = float(request.args.get('longitude'))

    weather_data = predict_weather(start_date, periods, latitude, longitude)

    json_result = json.dumps(weather_data.to_dict(orient='records'), default=str)

    return json_result


@app.route('/predict_days', methods=['GET'])
def predict_days():
    start_date = request.args.get('start_date')
    days = int(request.args.get('days', 1))
    latitude = float(request.args.get('latitude'))
    longitude = float(request.args.get('longitude'))

    periods = days * 24

    weather_data = predict_weather(start_date, periods, latitude, longitude)

    result_df = calculate_daily_medians(weather_data, days)

    json_result = json.dumps(result_df.to_dict(), default=str)

    return json_result


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
