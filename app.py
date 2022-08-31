import joblib
from flask import Flask, render_template, request

app = Flask(__name__)
model = joblib.load('models/model.h5')
scaler = joblib.load('models/scaler.h5')


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET'])
def get_predict():
    inp_data = [
        request.args.get("MinTemp"),
        request.args.get("MaxTemp"),
        request.args.get("Rainfall"),
        request.args.get("Sunshine"),
        request.args.get("WindGustSpeed"),
        request.args.get("Humidity9am"),
        request.args.get("Humidity3pm"),
        request.args.get("Pressure9am"),
        request.args.get("Temp9am")
    ]
    will_rain_tomorrow = model.predict(scaler.transform([inp_data]))[0]

    if will_rain_tomorrow == 0:
        will_rain_tomorrow = 'NO'
    elif will_rain_tomorrow == 1:
        will_rain_tomorrow = 'YES'

    return render_template("index.html", will_rain_tomorrow=will_rain_tomorrow)


if __name__ == '__main__':
    app.run(debug=True, host="127.0.0.22")
