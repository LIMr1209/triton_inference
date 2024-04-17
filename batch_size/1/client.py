import asyncio
from asyncer import asyncify

import tritonclient.http as httpclient
from tritonclient.utils import *

from anyio.lowlevel import RunVar
from anyio import CapacityLimiter


def send_request(x):
    model_name = "batch_size"

    with httpclient.InferenceServerClient("localhost:8000") as client:
        inputs = [
            httpclient.InferInput('input', [1, 1], "BYTES"),
        ]
        inputs[0].set_data_from_numpy(np.expand_dims(np.array([x], dtype=object), axis=0), binary_data=True)

        outputs = [
            httpclient.InferRequestedOutput("output", binary_data=True),
        ]

        response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)

        result = response.as_numpy('output')[0]
        print(x, result)



async def main():
    RunVar("_default_thread_limiter").set(CapacityLimiter(5000))
    tasks = [asyncify(send_request)(x) for x in range(1000)]
    await asyncio.gather(*tasks)

asyncio.run(main())
