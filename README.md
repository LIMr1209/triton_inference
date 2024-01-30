# 测试私有化部署

## 模型文件加解密
- 事先 设置一个固定字符串通过 md5 加密生成 AES key, 使用 AES key 通过 AES 对称加密算法 加密模型文件
- 运行时 通过同样的算法 解密模型文件 为 bytes
- pytorch 加载 bytes
- 这个固定字符串可以使用目标机器的 mac地址(全球唯一) 实现设备绑定
## Cython 编译
- 通过 Cython 编译项目所有 py 文件为 共享链接库 (windows pyd, linux so), 保持项目结构 
- 入口 manage.py 文件 不编译 
- 依赖现有 python 环境运行 manage.py 文件
- `python setup_main.py` 
- `python manage.py`
## codemeter 加密
- `python axprotector_build.py`
## 问题
- 需要测试性能
- 复杂Pytorch代码和环境未测试

# 推理服务

## Triton 自定义后端
- 编译后的so文件也可以使用
- 启动容器
```shell
docker run --runtime=nvidia --shm-size=1g --ulimit memlock=-1 -p 8000:8000 -p 8001:8001 -p 8002:8002 --ulimit stack=67108864 -ti nvcr.io/nvidia/tritonserver:23.11-py3
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

```



