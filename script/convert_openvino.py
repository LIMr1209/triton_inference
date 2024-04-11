import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openvino as ov
from torchvision.models.resnet import resnet152, ResNet152_Weights
import torch
input_dim = (64, 3, 116, 116)

dummy_input = torch.randn(input_dim)

model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
model.eval()
ov_model = ov.convert_model(model, example_input=torch.rand(1, 3, 224, 224))
ov.save_model(ov_model,  '../openvino_background/1/model.xml')


# import numpy as np
# core = ov.Core()
# model = core.read_model('../openvino_background/1/model.xml')
# compiled_model = core.compile_model(model, "CPU")
# for i in range(len(compiled_model.inputs)):
#     print(compiled_model.inputs[i].names)
# for i in range(len(compiled_model.outputs)):
#     print(compiled_model.outputs[i].names)
#
# input_dim = (64, 3, 128, 128)
#
# dummy_input = np.random.random(input_dim).astype(np.float32)
# infer_request = compiled_model.create_infer_request()
# output_tensor = infer_request.infer(inputs={"x": dummy_input})
# res = output_tensor.to_dict()
