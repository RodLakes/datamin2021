from flask import Flask, jsonify, request
from flask_cors import CORS
import random
from predict import predictPrice
from train import train
import os

# configuration
DEBUG = True

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

dfs = [
    {
        'LocationCod': '1',
        'MinTemp': 18,
        'MaxTemp': 25.5,
        'Rainfall': 0,
        'WindGustDirCod': 2,
        'WindGustSpeed': 44,
        'WindDir9amCod': 5,
        'WindDir3pmCod': 4,
        'WindSpeed9am': 20,
        'WindSpeed3pm': 24,
        'Humidity9am': 25,
        'Humidity3pm': 75,
        'Pressure9am': 50,
        'Pressure3pm': 45,
        'Temp9am': 15,
        'Temp3pm': 15,
        'RainTodayCod': 0,
        'RISK_MM': 'NO',

    }
]

# sanity check route
@app.route('/ping', methods=['GET'])
def ping_pong():
    return jsonify('pong!')

@app.route('/llueve', methods=['GET', 'POST'])
def all_dfs():
    response_object = {'status': 'success'}
    if request.method == 'POST':
        post_data = request.get_json()
        dfs.append({
            'Mes': post_data.get('Mes'),
            'LocationCod': post_data.get('LocationCod'),
            'MinTemp': post_data.get('MinTemp'),
            'MaxTemp': post_data.get('MaxTemp'),
            'Rainfall': post_data.get('Rainfall'),
            'WindGustDirCod': post_data.get('WindGustDirCod'),
            'WindDir9amCod': post_data.get('WindDir9amCod'),
            'WindDir3pmCod': post_data.get('WindDir3pmCod'),
            'WindSpeed9am': post_data.get('WindSpeed9am'),
            'WindSpeed3pm': post_data.get('WindSpeed3pm'),
            'Pressure9am': post_data.get('Pressure9am'),
            'Pressure3pm': post_data.get('Pressure3pm'),
            'Temp9am': post_data.get('Temp9am'),
            'Temp3pm': post_data.get('Temp3pm'),
            'RaintodayCod': post_data.get('RaintodayCod'),
            'RISK_MM': round(predictRisk(
                Mes=post_data.get('Mes'),
                LocationCod=post_data.get('LocationCod'),
                Mintemp=post_data.get('MinTemp'),
                MaxTemp=post_data.get('MaxTemp'),
                Rainfall=post_data.get('Rainfall'),
                WindGustDirCod=post_data.get('WinGustDirCod'),
                WindDir9amCod=post_data.get('WindDir9amCod'),
                WindDir3pmCod=post_data.get('WindDir3pmCod'),
                WindSpeed9am=post_data.get('WindSpeed9am'),
                WindSpeed3pm=post_data.get('WindSpeed3pm'),
                Pressure9am=post_data.get('Pressure9am'),
                Pressure3pm=post_data.get('Pressure3pm'),
                Temp9am=post_data.get('Temp9am'),
                Temp3pm=post_data.get('Temp3pm'),
                RaintodayCod=post_data.get('RaintodayCod'),
            )*1.01),  # To return from model
        })
        response_object['message'] = 'Pronostico!'
    else:
        response_object['llueve'] = dfs
    return jsonify(response_object)

@app.route('/train', methods=['GET'])
def train_model():
    response_object = {'status': 'success'}
    response_object['score'] = train()
    return jsonify(response_object)



if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))

    print("Starting app on port %d" % port)
    if(port!=5000):
        app.run(debug=False, port=port, host='0.0.0.0')
    else:
        app.run()
