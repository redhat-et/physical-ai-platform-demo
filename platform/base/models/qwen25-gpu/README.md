# Qwen2.5-14B-Instruct (GPU)

An instruction-tuned LLM served on an L40S GPU via stock vLLM (not vLLM-Omni, since
this is a plain text chat model with no multimodal/diffusion requirements). Sits
alongside `qwen25-cpu` in the catalog as a bigger, GPU-backed option. This is also
the model backing `platform_agent` itself (see `platform/base/agent/configmap.yaml`).

## Model Details

- **Model:** Qwen/Qwen2.5-14B-Instruct
- **Size:** 14.8B parameters
- **Hardware:** 1x NVIDIA L40S GPU (~35GB estimated VRAM at BF16, comfortably fits
  in 48GB with headroom for KV cache — no tensor parallelism needed)
- **Framework:** vLLM (`vllm/vllm-openai` image), tool calling enabled via the `hermes` parser

## Deployment

Deployed via KServe InferenceService with:

- **Runtime:** vllm-qwen25-gpu-runtime
- **minReplicas:** 0 — scales from zero via the `qwen25-gpu-http-scaler` HTTPScaledObject

## Testing

```bash
curl http://qwen25-gpu-predictor.physical-ai-models.svc.cluster.local/v1/models

curl http://qwen25-gpu-predictor.physical-ai-models.svc.cluster.local/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen25-gpu","messages":[{"role":"user","content":"Hello!"}]}'
```
