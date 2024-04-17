from datetime import datetime
import json
import os
import threading

import numpy as np
import triton_python_backend_utils as pb_utils

global_request_counter = 0


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])

    def execute(self, requests):
        responses = []
        global global_request_counter
        # batch_size 256
        print("batch_size", len(requests))

        thread_name = threading.current_thread().name
        thread_id = threading.current_thread().native_id
        active_count = threading.active_count()
        print(
            f"{datetime.now()} - {os.getpid()}-{thread_name}-{thread_id}-{active_count}: #{global_request_counter} processing, sleeping...")

        # 在这里，我们将收集所有请求的输入数据，以便一次性处理
        input_batch = []
        # 获取取消的请求
        cancel_index = []
        for i, request in enumerate(requests):
            # 请求取消
            if request.is_cancelled():
                cancel_index.append(cancel_index)
                continue

            global_request_counter += 1
            input_tensor = pb_utils.get_input_tensor_by_name(request, "input")
            input_batch.append(input_tensor.as_numpy())
        # 这里假设处理批次中所有数据
        processed_outputs = [input_data * 2 for input_data in input_batch]

        # 为每个输入生成对应的输出
        for output_data in processed_outputs:
            output_tensor = pb_utils.Tensor('output', np.array(output_data, dtype=object))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor]
            )
            responses.append(inference_response)
        # 处理取消的请求，并按顺序返回错误
        for index in cancel_index:
            responses.insert(index, pb_utils.InferenceResponse(
                error=pb_utils.TritonError("Message", pb_utils.TritonError.CANCELLED)))

        return responses

    def finalize(self):
        print("Cleaning up...")
