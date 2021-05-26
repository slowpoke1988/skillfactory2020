from flask import Flask, request
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
import pickle

app = Flask(__name__)
with open('model.pkl', 'rb') as input_file:
    regressor_from_file = pickle.load(input_file)
regressor_from_file
def model_predict(value):
    return regressor_from_file.predict(value)[0]

@app.route('/predict')

def predict_func():
    value = request.args.get('value')
    try:
        int(value)
    except ValueError:
        return f'{value} - не число. Ведите число.'
    else:
        value = np.array([int(value)])[:, np.newaxis]
        prediction = model_predict(value)
        return f'the result is {prediction}'  
if __name__ == '__main__':
    app.run('localhost', 5000)