from torchvision.models.resnet import resnet152, ResNet152_Weights
import torch
import numpy as np

model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
model.eval()

# import time
# start = time.time()
input_dim = (64, 3, 116, 116)

dummy_input = torch.randn(input_dim)

# output_dim = model(dummy_input)

# print(time.time()-start)

onnx_path = '../onnxruntime_background/1/model.onnx'

dynamic = {'INPUT0': {0: 'batch_size', 2: 'input_height', 3: 'input_width'}, 'OUTPUT0': {0: 'batch_size'}}

torch.onnx.export(model,
                  dummy_input,
                  onnx_path,
                  verbose=True,
                  input_names=['INPUT0'],
                  output_names=['OUTPUT0'],
                  dynamic_axes=dynamic,
                  opset_version=17)

# import onnxruntime as ort
#
# session = ort.InferenceSession(onnx_path)
# input_name = session.get_inputs()[0].name
#
# dummy_input = np.random.random(input_dim)
# b = dummy_input.astype(np.float32)
# outputs = session.run(None, {input_name: b})