# Infrastructure Services

## 1. MQTT & Redis Infrastructure
To start Mosquitto MQTT and Redis for telemetry and messaging:
```bash
docker compose -f docker/docker-compose.infra.yml up -d
```

## 2. Triton Inference Server
To run NVIDIA Triton Inference Server with local GPU support and mounted model repository:

### For Jetson / Tegra (igpu):
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

### For x86_64 / Discrete NVIDIA GPU:
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

### Verify Triton Endpoints:
- HTTP v2 Health: `curl -v http://localhost:8000/v2/health/ready`
- Model Metadata: `curl http://localhost:8000/v2/models/<model_name> | jq`
- gRPC Port: `localhost:8001`
- Metrics Port: `http://localhost:8002/metrics`
