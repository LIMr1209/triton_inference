# cython: language_level=3
import sys

import numpy as np
import tritonclient.http as httpclient
from PIL import Image
from tritonclient.utils import *


model_name = "torchscript_background"

with httpclient.InferenceServerClient("localhost:8000") as client:
    inputs = [httpclient.InferInput('INPUT0', [1, 3, 116, 116], "FP32")]
    img = Image.open("example.jpg").convert('RGB')
    b = np.expand_dims(np.array(img, dtype=np.float32), axis=-1).transpose()
    inputs[0].set_data_from_numpy(b)

    outputs = [
        httpclient.InferRequestedOutput("OUTPUT0"),
    ]

    response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)

    result = response.as_numpy('OUTPUT0')
    print(result.shape)
    with open("output.png", "wb") as f:
        f.write(result[0])
    sys.exit(0)
