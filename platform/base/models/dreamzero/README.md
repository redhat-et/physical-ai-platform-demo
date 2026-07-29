# DreamZero Model Deployment

DreamZero is a vision-language-action model for robotics tasks from GEAR-Dreams.

## Model Details

- **Model:** GEAR-Dreams/DreamZero-DROID
- **Type:** Multimodal diffusion model for robot control
- **Size:** 39GB (10 model shards)
- **GPU:** 1x NVIDIA GPU with 48GB+ VRAM required
- **Framework:** vLLM Omni

## Dataset Compatibility

DreamZero is inference-only on this platform — no fine-tuning recipe exists
here, so don't offer to fine-tune it. Because there's no recipe to check a
candidate dataset against, this section isn't formatted as the
Dimension/Priority checklist table used in pi05's README (see
`platform/base/models/pi05/README.md`) — it's purely training-data
provenance, for answering "what was this trained on" questions.

Trained on `GEAR-Dreams/DreamZero-DROID-Data`: 57,774 episodes (~131-145GB,
14.7M frames), a filtered DROID derivative with idle frames, non-annotated
episodes, and unsuccessful episodes removed. This is not the "~75k" figure
sometimes quoted for DROID — that figure is DROID's overall annotation
coverage, not this repo's episode count.

Uses relative joint positions as its action space, unlike the cartesian
encoding most raw LeRobot DROID ports expose.

`DreamZero-AgiBot` is a separate checkpoint trained on different data
(AgiBot G1 teleop) — don't conflate it with DreamZero-DROID when discussing
training data.

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
