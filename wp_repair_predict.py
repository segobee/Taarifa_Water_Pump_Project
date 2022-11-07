#!/usr/bin/env python
# coding: utf-8

import pickle
import numpy as np

from flask import Flask, request, jsonify

def predict_single(water_pump_info, dv, rf_model):
    X = dv.transform([water_pump_info]) 
    y_pred = rf_model.predict_proba(X)[:, 1]
    return y_pred[0]


with open('model_n_est=80.bin', 'rb') as f_in:
    dv, rf_model = pickle.load(f_in)

app = Flask('waterpump')

@app.route('/predict', methods=['POST'])
def predict():
    water_pump_info = request.get_json()

    prediction = predict_single(water_pump_info, dv, rf_model)
    wp_repair = prediction >= 0.5
    
    result = {
        'water_repair_probability': float(prediction),
        'wp_repair': bool(wp_repair),
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)