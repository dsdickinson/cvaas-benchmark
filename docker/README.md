# Infrastructure Services

## 1. MQTT & Redis Infrastructure
To start Mosquitto MQTT and Redis for telemetry and messaging:
```bash
docker compose -f docker/docker-compose.infra.yml up -d
```

## 2. Triton Inference Server Deployment

Triton can run either in **Remote GPU Mode** (NVIDIA TensorRT / CUDA) or **Remote CPU Mode** (OpenVINO / ONNX CPU).

### A. Remote GPU Mode (Jetson Orin / Tegra iGPU)
```bash
docker run -d --gpus=all --runtime=nvidia --rm \
  --name tritonserver \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  -v $(pwd)/models:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3-igpu \
  tritonserver --model-repository=/models --strict-model-config=false
```

### B. Remote GPU Mode (x86_64 Discrete NVIDIA GPU)
```bash
docker run -d --gpus=all --rm \
  --name tritonserver \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  -v $(pwd)/models:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models --strict-model-config=false
```

### C. Remote CPU Mode (x86_64 Server Without GPU)
```bash
docker run -d --rm \
  --name tritonserver \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  -v $(pwd)/models:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models --strict-model-config=false
```

---

## 3. Triton Model Configuration (`config.pbtxt`)

To specify whether Triton runs on GPU or CPU, configure the `instance_group` in your model directory (`models/<model_name>/config.pbtxt`):

### GPU Execution (`KIND_GPU`):
```protobuf
name: "object_detection"
platform: "tensorrt_plan" # or onnxruntime_onnx
max_batch_size: 8

instance_group [
  {
    kind: KIND_GPU
    count: 1
    gpus: [ 0 ]
  }
]
```

### CPU Execution (`KIND_CPU`):
```protobuf
name: "object_detection"
platform: "onnxruntime_onnx" # or openvino_onnx
max_batch_size: 8

instance_group [
  {
    kind: KIND_CPU
    count: 2
  }
]
```

---

## 4. Connecting Clients to Remote GPU vs CPU Triton

Both Python and Rust pipelines query and verify remote Triton instances automatically:

```bash
# Connect to Remote GPU Triton server via gRPC
python python/src/main.py --source video/intersection-night.mp4 --protocol grpc --server 192.168.1.80:8001 --gpu

# Connect to Remote CPU Triton server via HTTP
python python/src/main.py --source video/intersection-night.mp4 --protocol http --server 192.168.1.80:8000 --cpu
```

---

## 5. Verify Triton Endpoints:
- HTTP v2 Health: `curl -v http://localhost:8000/v2/health/ready`
- Model Metadata & Backend: `curl http://localhost:8000/v2/models/object_detection/config | jq`
- gRPC Port: `localhost:8001`
- Metrics Port: `http://localhost:8002/metrics`
