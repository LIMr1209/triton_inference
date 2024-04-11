import openvino as ov
from torchvision.models.resnet import resnet152, ResNet152_Weights
import torch
import numpy as np
input_dim = (64, 3, 116, 116)

dummy_input = torch.randn(input_dim)

model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
model.eval()
ov_model = ov.convert_model(model, example_input=torch.rand(1, 3, 224, 224))

# 2. Compile model from memory
core = ov.Core()
compiled_model = core.compile_model(ov_model)