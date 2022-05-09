# Serve model as a flask application

import pickle
import numpy as np
from flask import Flask, request
import json

model = None
app = Flask(__name__)


def load_model():
    global model
    # model variable refers to the global variable
    with open('uplift_model.sav', 'rb') as f:
        model = pickle.load(f)


@app.route('/')
def home_endpoint():
    return 'Hello World!'


@app.route('/predict', methods=['POST'])
def get_prediction():
    # Works only for a single sample
    if request.method == 'POST':
        datajs = request.get_json()  # Get data posted as a json
        #datadict = datajs.loads() #convert json object to dict
        datalist = datajs['data'] #extract values in list
        dataarr = np.array(datalist) #convert list to array
        #dataarr = dataarr.reshape(1,12)  # converts shape from (12,) to (1, 12)
        prediction = model.predict(dataarr)  # runs globally loaded model on the data
    return str(prediction[0])


if __name__ == '__main__':
    load_model()  # load model at the beginning once only
    app.run(host='0.0.0.0', port=12345)