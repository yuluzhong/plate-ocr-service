# 车牌检测与识别服务

使用 YOLO 定位车牌并进行透视纠偏，使用微调后的 PP-OCRv5 识别车牌文字，通过 FastAPI 提供 HTTP 接口。识别模型从本地加载。

## 文件

- `paddle_ocr_service.py`：OCR 服务入口。
- `util.py`：模型加载、透视纠偏及文字处理。
- `best1.pt`：车牌检测权重。
- `PP-OCRv5_server_rec_infer/`：微调后的文字识别模型，权重、配置和字典必须一起保留。
- `requirements.txt`：业务依赖的参考版本。
- `main_api.py`：可选的视频车辆处理入口，另需车辆检测权重 `yolov8n.pt`；单独运行 OCR 不需要启动它。

## 已运行环境

| 组件 | 版本 |
| --- | --- |
| 操作系统 | Linux x86_64，glibc 2.35 |
| Python | 3.10.20 |
| PyTorch | 2.1.0+cu118 |
| torchvision | 0.16.0+cu118 |
| PaddlePaddle GPU | 3.3.1，CUDA 11.8 |
| PaddleOCR / PaddleX | 3.5.0 / 3.5.1 |
| Ultralytics | 8.4.60 |
| NVIDIA 驱动 | 580.159.04 |

上述版本来自实际运行环境。当前组合已完成服务请求验证；以下安装步骤尚未在全新容器中完整复测，间接依赖未全部锁定。
本地微调模型包含 `linear_v2` 算子，PaddlePaddle 3.2.2 曾加载失败；请使用这里记录的 3.3.1。

## 安装

容器需配置 NVIDIA GPU 访问权限（例如 Docker 的 `--gpus all`）。从宿主机访问时映射 `9080:9080`。

在新的 Python 3.10 环境中安装。使用 Conda 时：

```bash
conda create -n plate_ocr python=3.10.20 pip -y
conda activate plate_ocr
python -m pip install --upgrade pip
```

Ubuntu/Debian 容器中的系统依赖：

```bash
apt-get update
apt-get install -y libgl1 libglib2.0-0 libgomp1
```

先分别安装两个 GPU 框架，再安装业务依赖，按以下顺序执行：

```bash
python -m pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 \
  --index-url https://download.pytorch.org/whl/cu118

python -m pip install paddlepaddle-gpu==3.3.1 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

python -m pip install -r requirements.txt
```

PyTorch 与 PaddlePaddle 共同使用部分 NVIDIA 运行库；若安装器报告版本冲突，应核对完整输出，不要直接忽略。当前已运行环境曾出现 Requests 和 Transformers 依赖警告，可选 Transformers 功能未作为本服务验证范围。

官方安装来源：[PyTorch](https://pytorch.org/get-started/previous-versions/)、[PaddlePaddle CUDA 11.8 包](https://www.paddlepaddle.org.cn/packages/stable/cu118/paddlepaddle-gpu/)。

## 启动

在项目目录执行：

```bash
CUDA_VISIBLE_DEVICES=0 PADDLE_DEVICE=gpu:0 python paddle_ocr_service.py --host 0.0.0.0 --port 9080
```

按实际 GPU 分配调整 `CUDA_VISIBLE_DEVICES`。首次收到有效识别请求时会初始化 PaddleOCR；HTTP 服务启动不代表 OCR 模型已加载成功。

接口文档：`http://127.0.0.1:9080/docs`。

- `POST /cci_ai/service/v1/plate_number`：请求体为图片二进制，Content-Type 使用 `application/octet-stream`。
- `POST /cci_ai/service/v1/plate_number_batch`：以 multipart/form-data 上传多个文件，字段名为 `files`。

识别结果位于 `result.item_list`，包含 `number`、`type`、`score`。当前代码中的 `type` 为空字符串。每张图片最多返回一个优先车牌结果。
