# cython: language_level=3
import sys

import tritonclient.http as httpclient
from tritonclient.utils import *


model_name = "tensorrt_background"

with httpclient.InferenceServerClient("localhost:8000") as client:
    inputs = [httpclient.InferInput('x', [64, 3, 512, 512], "FP32")]
    input_dim = (64, 3, 512, 512)

    dummy_input = np.random.random(input_dim)
    b = dummy_input.astype(np.float32)
    inputs[0].set_data_from_numpy(b)

    outputs = [
        httpclient.InferRequestedOutput("Result_7875"),
    ]

    response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)

    result = response.as_numpy('Result_7875')
    print(result.shape)
    sys.exit(0)
