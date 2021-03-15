# Flask server for model deployment

[Running locally](#running-sample-locally)  
[Using endpoint](#sending-a-request)

This project creates a Flask server with an API endpoint to get predictions from a TensorFlow model. It is optimized for models exported from Lobe but could be used with any TensorFlow models with some updates. 

Lobe has an endpoint built in that can be used while running the app and this endpoint works the same way. If your app works with the Lobe API, it will work with this sample just by updating the URL.

The code takes in a base64 image and returns an array of predictions and confidences. 


This sample is a quick way to build an endpoint for using your TensorFlow model.

The server code that defines endpoints is in `app.py`

The code for using your model including image pre-processing and output formatting for a prediction is in `tf_model_helper.py`

Swagger definition file in `swagger/` for reference


## Running sample locally

### Windows

Prerequisites for running on your machine
* Python 3.6 or 3.7

Create and activate a virtual environment  
`python -m venv .venv`  
`.venv/Scripts/activate`  
Install dependencies  
`python -m pip install --upgrade pip && pip install -r requirements.txt`

Run the server locally  
`python app.py`

### MacOs

Prerequisites for running on your machine

* Python 3.6 or 3.7

Create and activate a virtual environment
`python -m venv .venv`
`source .venv/bin/activate`
Install dependencies
`python -m pip install --upgrade pip && pip install -r requirements.txt`

Run the server
`python app.py` or

`export FLASK_APP=app.py`
`flask run`


## Sending a request

* post request to target url/predict  
  `{"inputs": {"Image": "<base64 image>"}}`
* successful prediction returns json  
  `{predictions: [{"predicted_class: confidence}]`
