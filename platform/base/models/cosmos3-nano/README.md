# Cosmos3-Nano Model Deployment

This directory contains the configuration to deploy NVIDIA's Cosmos3-Nano multimodal video generation model using vLLM omni runtime on OpenShift AI.

## Prerequisites

### 1. HuggingFace Token with Gated Model Access

You need a HuggingFace token with access to two gated repositories:

1. **Cosmos3-Nano model**: https://huggingface.co/nvidia/Cosmos3-Nano
2. **Cosmos3 Guardrails** (required by NVIDIA license): https://huggingface.co/nvidia/Cosmos-1.0-Guardrail

**To get access:**
1. Create a HuggingFace account at https://huggingface.co
2. Generate an access token at https://huggingface.co/settings/tokens
3. Visit both model pages above and accept the license agreements
4. Wait for approval (usually instant for NVIDIA models)

### 2. GPU Requirements

- **GPU**: NVIDIA L40S (48GB VRAM)
- **Memory**: 64-128GB RAM
- **Storage**: ~40GB for model weights

## Deployment

### One-Time Setup (Required Before First Deployment)

The Hugging Face secret only needs to be created **once** per namespace.

```bash
#    This secret persists and is reused by all deployments
oc create secret generic huggingface-token \
  --from-literal=HF_TOKEN=hf_your_token_here \
  -n physical-ai-models
```

**Note:** The secret persists in the namespace. You only need to recreate it if:
- The namespace is deleted
- The HuggingFace token expires or is revoked
- You want to rotate credentials

### Deploy with Kustomize

```bash
# From the repository root, install the shared runtime before the model.
oc apply -k platform/base/model-serving/vllm-omni
oc apply -k platform/base/models/cosmos3-nano
```

## How It Works

The upstream CUDA image supplies the Cosmos guardrail dependencies. Startup
does not install packages or request root. vLLM downloads the gated guardrail
model on initialization using `HF_TOKEN`; guardrails remain enabled. The server
uses `--enable-layerwise-offload` for the L40S and listens on port 8080.

## Generating Videos

### Text-to-Video (Async - Recommended)

For videos that take longer than 10 minutes to generate, use the async endpoint:

```bash
# Start generation
oc exec -n physical-ai-models <pod-name> -c kserve-container -- \
curl -X POST http://localhost:8080/v1/videos \
  -F "model=cosmos3-nano-isvc" \
  -F "prompt=A robot arm assembles circuit boards in a modern factory" \
  -F "negative_prompt=blurry, distorted, low quality, jittery, deformed" \
  -F "size=1280x720" \
  -F "num_frames=121" \
  -F "fps=24" \
  -F "num_inference_steps=25" \
  -F "guidance_scale=6.0" \
  -F "max_sequence_length=4096" \
  -F "flow_shift=10.0" \
  -F 'extra_params={"use_resolution_template":false,"use_duration_template":false,"guardrails":true}' \
  -F "seed=42"

# Get the video_id from the response, then check status
oc exec -n physical-ai-models <pod-name> -c kserve-container -- \
curl http://localhost:8080/v1/videos/<video_id>

# When completed, download the video
oc cp physical-ai-models/<pod-name>:/tmp/storage/<video_id>.mp4 ./output.mp4 -c kserve-container
```

### Text-to-Video (Sync - For Quick Videos)

For videos under 10 minutes generation time (≤20 inference steps):

```bash
oc exec -n physical-ai-models <pod-name> -c kserve-container -- \
curl -X POST http://localhost:8080/v1/videos/sync \
  -H "Accept: video/mp4" \
  -F "model=cosmos3-nano-isvc" \
  -F "prompt=A car hitting a snowbank while driving on the road" \
  -F "negative_prompt=blurry, distorted, low quality" \
  -F "size=1280x720" \
  -F "num_frames=121" \
  -F "fps=24" \
  -F "num_inference_steps=20" \
  -F "guidance_scale=6.0" \
  -F "flow_shift=10.0" \
  -o output.mp4
```

**Note:** Sync endpoint has a 600s (10 minute) timeout. For longer videos, use the async endpoint.

### Text-to-Image

```bash
oc exec -n physical-ai-models <pod-name> -c kserve-container -- \
curl -X POST http://localhost:8080/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cosmos3-nano-isvc",
    "prompt": "A photorealistic red sports car on a city street at golden hour",
    "negative_prompt": "blurry, distorted, low quality",
    "size": "1024x1024",
    "n": 1,
    "response_format": "b64_json",
    "num_inference_steps": 50,
    "guidance_scale": 7.0,
    "seed": 42
  }'
```

### Recommended Parameters

**For Video Generation:**
- **Resolution**: 1280x720 (default)
- **Frames**: 121 (5 seconds @ 24fps) or 189 (7.875 seconds @ 24fps)
- **Inference Steps**: 25 (~4 min generation) or 35 (~6 min generation)
- **Guidance Scale**: 6.0
- **Flow Shift**: 10.0
- **Use async endpoint** for steps > 20

**For Image Generation:**
- **Resolution**: 1024x1024
- **Inference Steps**: 50
- **Guidance Scale**: 7.0
- **Flow Shift**: 3.0

## Testing

Once deployed, check the pod status:

```bash
# Check pod is running
oc get pods -n physical-ai-models -l serving.kserve.io/inferenceservice=cosmos3-nano-isvc

# Check logs
oc logs -n physical-ai-models -l serving.kserve.io/inferenceservice=cosmos3-nano-isvc -c kserve-container

# Test the API
oc exec -n physical-ai-models <pod-name> -c kserve-container -- curl -s http://localhost:8080/v1/models
```

Expected output:
```json
{
  "object": "list",
  "data": [{
    "id": "cosmos3-nano-isvc",
    "object": "model",
    "owned_by": "vllm"
  }]
}
```

## Troubleshooting

### "401 Unauthorized" errors for Cosmos-1.0-Guardrail

Your HuggingFace token doesn't have access to the guardrail model:
1. Visit https://huggingface.co/nvidia/Cosmos-1.0-Guardrail
2. Accept the license agreement
3. Wait a few minutes for approval
4. Recreate the secret with the new token

### Model download stuck or slow

The Cosmos3-Nano model is 33GB. First-time download can take 10-30 minutes depending on network speed. Check progress:
```bash
oc exec -n physical-ai-models <pod-name> -c storage-initializer -- du -sh /mnt/models
```

### Video generation timeout (504 Gateway Timeout)

The sync endpoint has a 600s (10 minute) timeout. Solutions:
1. **Use async endpoint** (recommended) - no timeout
2. **Reduce inference steps** to 20 or less
3. **Reduce num_frames** to 89 (shorter video)

### Video generation too slow

Each inference step takes ~10-20 seconds. To speed up:
- Reduce `num_inference_steps` (25 is good, 35 is high quality but slow)
- Reduce `num_frames` (fewer frames = faster generation)
- Use smaller resolution (1280x720 is default, could use 960x544)

## Model Information

- **Name**: Cosmos3-Nano
- **Type**: Multimodal world foundation model (video generation, image generation)
- **Size**: 33GB
- **Modalities**: Text-to-Video, Text-to-Image, Image-to-Video, Video-to-Video
- **License**: NVIDIA Open Model License (requires guardrails)
- **HuggingFace**: https://huggingface.co/nvidia/Cosmos3-Nano
- **Documentation**: https://github.com/vllm-project/vllm-omni/blob/main/recipes/cosmos3/Cosmos3-Nano.md
