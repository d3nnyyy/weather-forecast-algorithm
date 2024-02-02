"""
This Flask application defines a /predict endpoint that takes
start_date and optional periods parameters from a GET request,
uses the predict_weather function to generate weather forecasts,
converts the results to JSON, and returns the predictions when the script is run.
"""

import json

from flask import Flask, request
from flask_cors import CORS

from tools import predict_weather

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


if __name__ == '__main__':
    app.run(debug=True)
