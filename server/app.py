from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import pickle
import os

if not os.path.isfile('rf.sav'):
    print('Model not found! Generate the ML model before running the server')
    exit(1)

model = pickle.load(open('rf.sav', 'rb'))

app = Flask(__name__)
app.config.from_object(__name__)

CORS(app, resources={r'/*': {'origins': '*'}})

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify('Server running!')

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    bruises = request.form.get("bruises")
    if bruises == 'on':
        bruises = 1
    else:
        bruises = 0

    mushroom_data = [[
        0,
        bruises,
        request.form.get("odor", type=int),
        request.form.get("gill-spacing", type=int),
        request.form.get("gill-size", type=int),
        request.form.get("gill-color", type=int),
        request.form.get("stalk-shape", type=int),
        request.form.get("stalk-root", type=int),
        request.form.get("stalk-surface-above-ring", type=int),
        request.form.get("stalk-surface-below-ring", type=int),
        request.form.get("stalk-color-above-ring", type=int),
        request.form.get("stalk-color-below-ring", type=int),
        request.form.get("ring-type", type=int),
        request.form.get("spore-print-color", type=int),
        request.form.get("population", type=int),
        request.form.get("habitat", type=int),
    ]]

    print(mushroom_data)

    pred = model.predict(mushroom_data)

    print(pred)

    result = pred[0]
    
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run()