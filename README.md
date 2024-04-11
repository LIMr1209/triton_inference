# 推理服务
## 安装依赖
```shell
pip install -r requirements.txt
```
## 下载docker 镜像
```shell
docker pull nvcr.io/nvidia/tritonserver:24.03-py3
docker run --runtime=nvidia --shm-size=1g --ulimit memlock=-1 -p 8000:8000 -p 8001:8001 -p 8002:8002 --ulimit stack=67108864 -ti nvcr.io/nvidia/tritonserver:23.11-py3
```
## Triton 自定义后端
- 编译后的so文件也可以使用
- 启动容器
```shell
git clone https://github.com/triton-inference-server/python_backend -b r23.11
cd python_backend
```
- 拷贝代码
```shell
docker cp custom_background fbc5c3a35ffd:/opt/tritonserver/python_backend/models/
```
- 启动服务
```shell
tritonserver --model-repository `pwd`/models --load-model=* --model-control-mode=explicit
```
- 测试
```shell
pip install tritonclient[all]
python3 client.py
```
## Triton TorchScript 后端
- 转换模型
```shell
cd script && python conver_torchscript.py
```
- 拷贝代码
```shell
docker cp torchscript_background e71311bd5529:/opt/tritonserver/models/
```
- 启动服务
```shell
tritonserver --model-repository `pwd`/models --load-model=* --model-control-mode=explicit
```
- 测试
```shell
pip install tritonclient[all]
python3 client.py
```


