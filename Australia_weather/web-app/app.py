import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("weather_predict.pkl", "rb"))

@app.route("/")
def Home():
	return render_template("index.html")


@app.route("/predict", methods = ["POST"])
def predict():
    location = request.form.get('Location')
    mintemp = float(request.form.get('Min Tempurature'))
    maxtemp = float(request.form.get('Max Tempurature'))
    rainfall = float(request.form.get('Rainfall'))
    windgustdir = request.form.get('Wind Direction')
    windgustspeed = float(request.form.get('Wind Speed'))
    winddir9am = request.form.get('Wind Direction at 9 AM')
    winddir3pm = request.form.get('Wind Direction at 3 PM')
    windspeed9am = float(request.form.get('Wind Speed at 9 AM'))
    windspeed3pm = float(request.form.get('Wind Speed at 3 PM'))
    humidity9am = float(request.form.get('Humidity at 9 AM'))
    humidity3pm = float(request.form.get('Humidity at 3 PM'))
    pressure9am = float(request.form.get('Pressure at 9 AM'))
    pressure3pm = float(request.form.get('Pressure at 3 PM'))
    temp9am = float(request.form.get('Tempurature at 9 AM'))
    temp3pm = float(request.form.get('Tempurature at 3 PM'))
    raintoday = request.form.get('Rain Today')
    final_features = pd.DataFrame({'location': {0: location},
                                    'mintemp': {0: mintemp},
                                    'maxtemp': {0: maxtemp},
                                    'rainfall': {0: rainfall},
                                    'windgustdir': {0: windgustdir},
                                    'windgustspeed': {0: windgustspeed},
                                    'winddir9am': {0: winddir9am},
                                    'winddir3pm': {0: winddir3pm},
                                    'windspeed9am': {0: windspeed9am},
                                    'windspeed3pm': {0: windspeed3pm},
                                    'humidity9am': {0: humidity9am},
                                    'humidity3pm': {0: humidity3pm},
                                    'pressure9am': {0: pressure9am},
                                    'pressure3pm': {0: pressure3pm},
                                    'temp9am': {0: temp9am},
                                    'temp3pm': {0: temp3pm},
                                    'raintoday': {0: raintoday}})    
    prediction = model.predict(final_features)
    return render_template("index.html", prediction_text = "Tomorrow will be {}".format('rainy' if prediction == 0 else 'sunny'))


if __name__  == "__main__":
	app.run(debug= True)
