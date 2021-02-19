from tf_example import Model

import os
import io
import re
import base64
from PIL import Image
from flask import Flask, request

app = Flask(__name__)

# Path to signature.json and model file
ASSETS_PATH = os.path.join(".", "./assets")
MODEL = Model(ASSETS_PATH)
MODEL.load()


@app.route('/predict', methods=["POST"])
def predict_image():
    req = request.get_json(force=True)
    image = _process_base64(req)
    result = MODEL.predict(image)
    return {"outputs": result }

def _process_base64(json_data):
    image_data = json_data.get("inputs").get("Image")
    image_data = re.sub(r"^data:image/.+;base64,", "", image_data)
    image_base64 = bytearray(image_data, "utf8")
    image = base64.decodebytes(image_base64)
    return Image.open(io.BytesIO(image))


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)