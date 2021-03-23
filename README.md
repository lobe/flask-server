<div style="text-align:center"><img src="https://github.com/lobe/web-bootstrap/blob/master/assets/header.jpg" /></div>

[Lobe](http://lobe.ai/) is a free, easy to use app that has everything you need to bring your machine learning ideas to life. This Flask starter project creates a REST API to get predictions from a TensorFlow model on your projects or apps. To start using it, follow the instructions below:

## Get Started

1. Clone, fork or download the project on your computer to get started. You'll need Python 3.6 or 3.7 to run this starter project as well.

#### Windows

2. Create and activate a virtual environment  
```python
python -m venv .venv
.venv/Scripts/activate
```

3. Install dependencies  
```python
python -m pip install --upgrade pip && pip install -r requirements.txt
```

4. Run the server locally  
```python
python app.py
```

#### macOS

2. Create and activate a virtual environment
```python
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies
```python
python -m pip install --upgrade pip && pip install -r requirements.txt
```

4. Run the server
```python
python app.py
# or
export FLASK_APP=app.py
flask run
```


### Sending a request

1. Perform a post request to the target url/predict with your base64 image.
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
      "predicted_class": "confidence"
    }
  ]
}
```

## Additional Information

The Flask starter project is optimized for models exported from Lobe but could be used with any TensorFlow models with some small updates.

Lobe has an endpoint built in called Lobe Connect that can be used while running the app and this starter project works the same way. If your app works with Lobe Connect, it will work with this starter project just by updating the URL.

The code takes in a `base64` image and returns an array of predictions and confidences. The server code that defines endpoints is in `app.py`. And the code for using your model including image pre-processing and output formatting for a prediction is in `tf_model_helper.py`. For reference, the Swagger definition file lives in `swagger/`.

## Contributing

GitHub Issues are for reporting bugs, discussing features and general feedback on the iOS starter project. Be sure to check our documentation, FAQ and past issues before opening any new ones.

To share your project, get feedback on it, and learn more about Lobe, please visit our community on [Reddit](https://www.reddit.com/r/Lobe/). We look forward to seeing the amazing projects that can be built, when machine learning is made accessible to you.
