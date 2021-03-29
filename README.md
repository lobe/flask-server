<div style="text-align:center"><img src="https://github.com/lobe/flask-server/blob/main/assets/header.jpg" /></div><br>

[Lobe](http://lobe.ai/) is a free, easy to use app that has everything you need to bring your machine learning ideas to life. This Flask starter project creates a REST API to get predictions from a TensorFlow model on your projects or apps. To start using it, follow the instructions below:

## Get Started

1. Clone or download the project on your computer to get started. You'll need Python 3.6 or 3.7 to run this starter project as well.

2. Move the model, weights and signature.json file exported from Lobe to the `/model` directory

#### Windows

3. Create and activate a virtual environment  
```python
python -m venv .venv
.venv\Scripts\activate
```

4. Install dependencies  
```python
python -m pip install --upgrade pip && pip install -r requirements.txt
```

5. Run the server locally  
```python
python app.py
```

#### macOS

3. Create and activate a virtual environment
```python
python -m venv .venv
source .venv/bin/activate
```

4. Install dependencies
```python
python -m pip install --upgrade pip && pip install -r requirements.txt
```

5. Run the server
```python
python app.py
# or
export FLASK_APP=app.py
flask run
```
#### Deploy to Azure App Service

1. Have version 2.0.80 or higher of [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli) installed.  
   `az --version`
2. Log in by running this command and following prompts   
`az login`
3. Deploy to the cloud!  
`az webapp up --sku B1 --name <your unique app name>`

Azure documentation is available if you run into issues. 
This [quick start](https://docs.microsoft.com/en-us/azure/app-service/quickstart-python?toc=%2Fpython%2Fazure%2FTOC.json&tabs=bash&pivots=python-framework-flask) is a good starting point.


### Sending a request

1. Perform a post request to the target url/predict with your base64 image. Refer to `testing.py` for getting started sending requsts to the server.
```JSON
{
  "image": "<base64 image>"
}
```

2. Successful requests return JSON with the confidences of your predictions.
```JSON
{
  "predictions": [
    {
      "predicted_label": 1.0
    },
    {
      "another_label": 0.0
    }
  ]
}
```

## Additional Information

The Flask starter project is optimized for models exported from Lobe but could be used with any TensorFlow models with some small updates.

Lobe has an endpoint built in called Lobe Connect that can be used while running the app and this starter project works the same way. If your app works with Lobe Connect, it will work with this starter project just by updating the URL.

The code takes in a `base64` image and returns an array of predictions and confidences. The server code that defines endpoints is in `app.py`. And the code for using your model including image pre-processing and output formatting for a prediction is in `tf_model_helper.py`. For reference, the Swagger definition file lives in `swagger/`.

## Contributing

GitHub Issues are for reporting bugs, discussing features and general feedback on the Flask starter project. Be sure to check our documentation, FAQ and past issues before opening any new ones.

To share your project, get feedback on it, and learn more about Lobe, please visit our community on [Reddit](https://www.reddit.com/r/Lobe/). We look forward to seeing the amazing projects that can be built, when machine learning is made accessible to you.
