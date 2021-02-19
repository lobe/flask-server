# Flask server for model deployment

Currently covered is deploying to Azure Web Service and running locally

## Running sample locally

* Python version >= 3.6
  * python -m venv .venv
  * source .venv/bin/activate
  * python -m pip install --upgrade pip && pip install -r requirements.txt
  * Run server (python app.py; export FLASK_APP=app.py, flask run; gunicorn app:app)
  * Terminal output shows url to target
* post request to target url/predict
  `{"inputs": {"Image": "<base64 image>"}}`
* successful prediction returns json

```json
{
  "outputs": {
    "Labels": [
      [
        "Apples", 
        1.0
      ], 
      [
        "Bananas", 
        0.0
      ], 
      [
        "Oranges", 
        0.0
      ], 
      [
        "Strawberries", 
        0.0
      ]
    ], 
    "Prediction": "Apples"
  }
}
```

## Azure App Service

* There is an [existing quick start](https://docs.microsoft.com/en-us/azure/app-service/quickstart-python?toc=%2Fpython%2Fazure%2FTOC.json&tabs=bash&pivots=python-framework-flask) that is up to date and helpful
  * Assumes Azure subscription and Python version >= 3.6
  * Uses Azure CLI - 3 commands max to deploy app
    1. az --version (ensure up to date resources)
    2. az login
    3. az webapp up --sku B1 --name app-name
  * Includes instructions for running locally
