import time

import numpy as np
import onnx
import onnxruntime as rt
import torch
from torchvision.models.resnet import resnet152, ResNet152_Weights
import tensorrt as trt
from cuda import cuda


def trt_inference(engine, context, data):
    nInput = []
    nOutput = []
    for i in range(engine.num_bindings):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            nInput.append(name)
        elif mode == trt.TensorIOMode.OUTPUT:
            nOutput.append(name)
    print('nInput:', nInput)
    print('nOutput:', nOutput)

    for i in nInput:
        print("Bind[%s]:i[%s]->" % (i, i), engine.get_tensor_dtype(i), engine.get_tensor_shape(i),
              context.get_tensor_shape(i))
    for i in nOutput:
        print("Bind[%s]:o[%s]->" % (i, i), engine.get_tensor_dtype(i), engine.get_tensor_shape(i),
              context.get_tensor_shape(i))

    bufferH = []
    bufferH.append(np.ascontiguousarray(data.reshape(-1)))

    for i in nOutput:
        bufferH.append(np.empty(context.get_tensor_shape(i), dtype=trt.nptype(engine.get_tensor_dtype(i))))

    bufferD = []
    for i in bufferH:
        bufferD.append(cuda.cuMemAlloc(i.nbytes)[1])

    for i in range(len(nInput)):
        cuda.cuMemcpyHtoD(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes)

    context.execute_v2(bufferD)

    for i in range(len(nInput), len(nInput) + len(nOutput)):
        cuda.cuMemcpyDtoH(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes)

    for b in bufferD:
        cuda.cuMemFree(b)

    return bufferH


model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
model.eval()

input_dim = (64, 3, 128, 128)

dummy_input = np.random.random(input_dim).astype(np.float32)

onnx_path = '../onnxruntime_background/1/model.onnx'
tensorrt_engine_path = "../tensorrt_background/1/model.plan"

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
# 为了使用 ONNX 解析器导入模型，需要EXPLICIT_BATCH标志。有关详细信息，请参阅显式与隐式批处理部分。
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
# 　# 这个接口有很多属性，你可以设置这些属性来控制 TensorRT 如何优化网络。一个重要的属性是最大工作空间大小。层实现通常需要一个临时工作空间，并且此参数限制了网络中任何层可以使用的最大大小。如果提供的工作空间不足，TensorRT 可能无法找到层的实现：
# tensorrt 7.x
#　config.max_workspace_size = 1 << 20
# tensorrt 8.x
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20)  # 1 MiB
# config.set_flag(trt.BuilderFlag.FP16)
parser = trt.OnnxParser(network, logger)
with open(onnx_path, "rb") as model:
    if not parser.parse(model.read()):
        print("Failed parsing .onnx file!")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        exit()
    print("Succeeded parsing .onnx file!")
inputTensor = network.get_input(0)
print('inputTensor.name:', inputTensor.name)
profile.set_shape(inputTensor.name, (1, 3, 128, 128), (32, 3, 256, 256), (64, 3, 512, 512))
config.add_optimization_profile(profile)
engineString = builder.build_serialized_network(network, config)
if engineString == None:
    print("Failed building engine!")
    exit()
print("Succeeded building engine!")
with open(tensorrt_engine_path, "wb") as f:
    f.write(engineString)

# Read the engine from the file and deserialize
with open(tensorrt_engine_path, "rb") as f, trt.Runtime(logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

# TensorRT inference
context.set_input_shape("INPUT0", (64, 3, 128, 128))

trt_start_time = time.time()
trt_outputs = trt_inference(engine, context, dummy_input)
trt_outputs = np.array(trt_outputs[1]).reshape(64, -1)
trt_end_time = time.time()

# ONNX inference
onnx_model = onnx.load(onnx_path)
sess = rt.InferenceSession(onnx_path)

input_all = [node.name for node in onnx_model.graph.input]
input_initializer = [
    node.name for node in onnx_model.graph.initializer
]
net_feed_input = list(set(input_all) - set(input_initializer))
assert len(net_feed_input) == 1

sess_input = sess.get_inputs()[0].name
sess_output = sess.get_outputs()[0].name

onnx_start_time = time.time()
onnx_result = sess.run([sess_output], {sess_input: dummy_input})[0]
onnx_end_time = time.time()

model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1).to("cuda:0")
model.eval()
img_resize_torch = torch.from_numpy(dummy_input).to("cuda:0")
torch_start_time = time.time()
pytorch_result = model(img_resize_torch)
torch_end_time = time.time()
pytorch_result = pytorch_result.detach().cpu().numpy()

print('--tensorrt--')
print(trt_outputs.shape)
print(trt_outputs[0][:10])
print(np.argmax(trt_outputs, axis=1))
print('Time: ', trt_end_time - trt_start_time)

print('--onnx--')
print(onnx_result.shape)
print(onnx_result[0][:10])
print(np.argmax(onnx_result, axis=1))
print('Time: ', onnx_end_time - onnx_start_time)

print('--pytorch--')
print(pytorch_result.shape)
print(pytorch_result[0][:10])
print(np.argmax(pytorch_result, axis=1))
print('Time: ', torch_end_time - torch_start_time)
