#  -------------------------------------------------------------
#   Copyright (c) Microsoft Corporation.  All rights reserved.
#  -------------------------------------------------------------
"""
Skeleton code showing how to load and run the TensorFlow SavedModel export package from Lobe.
"""
import os
import json
import numpy as np
from threading import Lock

# printing only warnings and error messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

try:
    import tensorflow as tf
    from PIL import Image
except ImportError:
    raise ImportError("ERROR: Failed to import libraries. Please refer to READEME.md file\n")

EXPORT_MODEL_VERSION = 1

class TFModel:
    
    def __init__(self, model_dir) -> None:
        # make sure our exported SavedModel folder exists
        self.model_path = os.path.realpath(model_dir)
        if not os.path.exists(self.model_path):
            raise ValueError(f"Exported model folder doesn't exist {model_dir}")
        # load our signature json file, this shows us the model inputs and outputs
        # you should open this file and take a look at the inputs/outputs to see their data types, shapes, and names
        with open(os.path.join(self.model_path, "signature.json"), "r") as f:
            self.signature = json.load(f)
        self.inputs = self.signature.get("inputs")
        self.outputs = self.signature.get("outputs")
        self.labels = self.signature.get("classes").get("Label")

        self.lock = Lock()

        # loading the saved model
        self.model = tf.saved_model.load(tags=self.signature.get("tags"), export_dir=self.model_dir)
        self.predict_fn = self.model.signatures["serving_default"]

        # check whether Lobe model in the app is the latest
        version = self.signature.get("export_model_version")
        if version is None or version != EXPORT_MODEL_VERSION:
            print(
                f"The model version you are using for this starter project may not be compatible with our supported versions. \
                    Please update both this starter project and Lobe to the latest version, and try exporting your model again. \
                        If the issue persists, please contact us at lobesupport@microsoft.com"
            )

    def predict(self, image: Image.Image) -> dict:
        # pre-processing the image before passing to model
        image = self.process_image(image, self.inputs.get("Image").get("shape"))

        with self.lock:
            # create the feed dictionary that is the input to the model
            feed_dict = {}
            # first, add our image to the dictionary (comes from our signature.json file)
            feed_dict[list(self.inputs.keys())[0]] = tf.convert_to_tensor(image)
            # run the model!
            outputs = self.predict_fn(**feed_dict)
            # return the processed output
            return self.process_output(outputs)

    def process_image(self, image, input_shape) -> np.ndarray:
        """
        Given a PIL Image, center square crop and resize to fit the expected model input, and convert from [0,255] to [0,1] values.
        """
        width, height = image.size
        # ensure image type is compatible with model and convert if not
        if image.mode != "RGB":
            image = image.convert("RGB")
        # center crop image (you can substitute any other method to make a square image, such as just resizing or padding edges with 0)
        if width != height:
            square_size = min(width, height)
            left = (width - square_size) / 2
            top = (height - square_size) / 2
            right = (width + square_size) / 2
            bottom = (height + square_size) / 2
            # Crop the center of the image
            image = image.crop((left, top, right, bottom))
        # now the image is square, resize it to be the right shape for the model input
        input_width, input_height = input_shape[1:3]
        if image.width != input_width or image.height != input_height:
            image = image.resize((input_width, input_height))

        # make 0-1 float instead of 0-255 int (that PIL Image loads by default)
        image = np.asarray(image) / 255.0
        # format input as model expects
        return np.expand_dims(image, axis=0).astype(np.float32)

    def process_output(self, fetches, outputs) -> dict:
        # do a bit of postprocessing
        out_keys = ["label", "confidence"]
        results = {}
        # since we actually ran on a batch of size 1, index out the items from the returned numpy arrays
        for key, tf_val in outputs.items():
            val = tf_val.numpy().tolist()[0]
            if isinstance(val, bytes):
                val = val.decode()
            results[key] = val
        confs = results["Confidences"]
        output = [dict(zip(out_keys, group)) for group in zip(self.labels, confs)]
        sorted_output = {"predictions": sorted(output, key=lambda k: k["confidence"], reverse=True)}
        return sorted_output

