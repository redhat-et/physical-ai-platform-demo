# DreamZero Model Deployment

DreamZero is a vision-language-action model for robotics tasks from GEAR-Dreams.

## Model Details

- **Model:** GEAR-Dreams/DreamZero-DROID
- **Type:** Multimodal diffusion model for robot control
- **Size:** 39GB (10 model shards)
- **GPU:** 1x NVIDIA GPU with 48GB+ VRAM required
- **Framework:** vLLM Omni

## Deployment

This model is deployed via KServe InferenceService with:
- **Runtime:** vllm-omni-runtime
- **GPU:** 1 GPU
- **Memory:** 32-64 GiB
- **Storage:** 100GB PVC for model caching

## Testing

### Prerequisites

1. **Install dependencies:**
```bash
pip install opencv-python openpi-client
```

2. **Download test assets:**
```bash
hf download YangshenDeng/vllm-omni-dreamzero-assets --repo-type dataset --local-dir ~/dreamzero-test/assets/
```

### Run Tests

1. **Port forward to the service:**
```bash
oc port-forward -n physical-ai-models svc/dreamzero-predictor 8000:8080
```

2. **Run OpenPI client:**
```bash
cd ~/dreamzero-test
python dreamzero_client.py --host 127.0.0.1 --port 8000 --video-dir assets/
```

## API Endpoint

Once deployed, the model is available at:
- **Internal:** `http://dreamzero-predictor.physical-ai-models.svc.cluster.local:8080`
- **External (via port-forward):** `http://localhost:8000`

### API Routes

- `/v1/realtime/robot/openpi` - OpenPI protocol WebSocket endpoint
- `/openapi.json` - OpenAPI schema
- `/docs` - Interactive API documentation
- `/metrics` - Prometheus metrics

## Resource Usage

- **GPU Memory:** ~43 GiB
- **System Memory:** 32-64 GiB
- **Storage:** 50GB (39GB model + overhead)
- **Load Time:** ~11 seconds (with cached model)
- **First Download:** ~10-15 minutes

## Multi-GPU Configuration

For better performance, you can scale to multiple GPUs:

### 2 GPUs (Tensor Parallelism)
Edit InferenceService to request 2 GPUs and update the runtime args to include:
```yaml
- --tensor-parallel-size=2
- --stage-overrides={"0":{"devices":"0,1"}}
```

### 4 GPUs (TP=2 + CFG=2)
Request 4 GPUs and update args:
```yaml
- --tensor-parallel-size=2
- --cfg-parallel-size=2
- --stage-overrides={"0":{"devices":"0,1,2,3"}}
```

## Troubleshooting

### Model Not Loading

Check pod logs:
```bash
oc logs -n physical-ai-models -l serving.kserve.io/inferenceservice=dreamzero -c kserve-container
```

### Download Issues

The model downloads on first startup. If it fails:
1. Check internet connectivity from the pod
2. Verify HF_HUB_OFFLINE is not set
3. Increase timeout settings

### Client Errors

If you get `TypeError: can't convert np.ndarray of type numpy.object_`:
- This is a known client data format issue
- Use the official vLLM example clients from the vLLM repository
- Ensure video frames are in the correct numpy dtype (uint8, not object)

## References

- [DreamZero Model Card](https://huggingface.co/GEAR-Dreams/DreamZero-DROID)
- [vLLM Omni Documentation](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/examples/online_serving/dreamzero/)
- [OpenPI Protocol](https://github.com/physical-intelligence/openpi)
