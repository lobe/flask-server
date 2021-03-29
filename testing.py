#  -------------------------------------------------------------
#   Copyright (c) Microsoft Corporation.  All rights reserved.
#  -------------------------------------------------------------

import base64
import requests


# Save string of image file path below
img_filepath = "<path/to/image.jpg>"

# Create base64 encoded string
with open(img_filepath, "rb") as f:
    image_string = base64.b64encode(f.read()).decode("utf-8")

# Get response from POST request
# Update the URL as needed
response = requests.post(
    url="http://localhost:5000/predict",
    json={"image": image_string},
)
data = response.json()
top_prediction = data["predictions"][0]

# Print the top predicted label and its confidence
print("predicted label:\t{}\nconfidence:\t\t{}"
      .format(top_prediction["label"], top_prediction["confidence"]))
