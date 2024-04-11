# 推理服务

## 下载docker 镜像
```shell
docker pull nvcr.io/nvidia/tritonserver:24.03-py3
docker run --runtime=nvidia --shm-size=1g --ulimit memlock=-1 -p 8000:8000 -p 8001:8001 -p 8002:8002 --ulimit stack=67108864 -ti nvcr.io/nvidia/tritonserver:23.11-py3
```

## 安装依赖
```shell
pip install -r requirements.txt
mkdir /opt/tritonserver/models
```

## Triton 自定义后端
- cython编译后的so文件也可以使用
- 拷贝代码
```shell
cp  -r /opt/triton_inference/custom_background /opt/tritonserver/models
```
- 启动服务
```shell
tritonserver --model-repository `pwd`/models --load-model=* --model-control-mode=explicit
```
- 测试
```shell
cd /opt/tritonserver/models/custom_background/1 && python3 client.py
```

## Triton TorchScript 后端
- 转换模型
```shell
cd script && python3 convert_torchscript.py
```
- 拷贝代码
```shell
cp  -r /opt/triton_inference/torchscript_background /opt/tritonserver/models
```
- 启动服务
```shell
tritonserver --model-repository `pwd`/models --load-model=* --model-control-mode=explicit
```
- 测试
```shell
cd /opt/tritonserver/models/torchscript_background/1 && python3 client.py
```

## Triton onnxruntime 后端
- 转换模型
```shell
cd script && python3 convert_onnx.py
```
- 拷贝代码
```shell
cp  -r /opt/triton_inference/onnxruntime_background /opt/tritonserver/models
```
- 启动服务
```shell
tritonserver --model-repository `pwd`/models --load-model=* --model-control-mode=explicit
```
- 测试
```shell
cd /opt/tritonserver/models/torchscript_background/1 && python3 client.py
```

## Triton TensorRT 后端
- tensorrt 10.0.0b6 有问题 num_bindings 未找到  正式版本 8.6.1, `ImportError: libcudnn.so.8` 不兼容 当前 cuda 12 cudnn 9.0.0
- 现阶段解决办法 nvcr.io/nvidia/tensorrt:24.03-py3 中的 tensorrt==8.6.3 正式版本(未发布到python 官方源) 拷贝到当前容器 
- 转换模型
```shell
cd script && python3 convert_tensorrt.py
```
- 拷贝代码
```shell
cp  -r /opt/triton_inference/tensorrt_background /opt/tritonserver/models
```
- 启动服务
```shell
tritonserver --model-repository `pwd`/models --load-model=* --model-control-mode=explicit
```
- 测试
```shell
cd /opt/tritonserver/models/tensorrt_background/1 && python3 client.py
```

## Triton openvino 后端
- 转换模型
- 好像只支持英特尔自家GPU
```shell
cd script && python3 convert_openvino.py
```
- 拷贝代码
```shell
cp  -r /opt/triton_inference/openvino_background /opt/tritonserver/models
```
- 启动服务
```shell
tritonserver --model-repository `pwd`/models --load-model=* --model-control-mode=explicit
```
- 测试
```shell
cd /opt/tritonserver/models/openvino_background/1 && python3 client.py
```
- 未通过有问题 `config.pbtxt`