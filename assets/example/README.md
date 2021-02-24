# Getting Started - TensorFlow

In the main export directory there is the exported model and a directory named variables that make up the Tensorflow model. There is also a file named signature.json which contains information about your Lobe project. With these, you are ready to use your model! If you want to see an example of how to use this model, there are instructions below for running a quick test script.

## Example Contents

`signature.json` is created by Lobe and contains information about the model such as label names and the image size and shape the model expects.

`tf_example.py` is a simple script to quickly test your exported model. It takes a path to an image on your file system, prepares the image and returns the predicted class and confidence level.

`requirements.txt` is where the Python libraries and version information required to run the script are found.

## Run Example

You will need Python 3.6 and the path to an image on your machine to test.

Create a virtual environment

`python -m venv tf-venv`

Activate the virtual environment

macOS `source tf-venv/bin/activate`

Windows `tf-venv/Scripts/activate`

Install the the dependencies for the example

`python -m pip install --upgrade pip && pip install -r requirements.txt`

Run the example and see the model output

`python tf_example.py path/to/image/for/testing`
