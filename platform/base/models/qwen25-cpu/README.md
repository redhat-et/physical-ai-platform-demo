# Qwen2.5-1.5B-Instruct (CPU)

A small instruction-tuned LLM served on CPU-only nodes via vLLM's CPU backend, for
environments/experiments where GPU capacity isn't available or needed.

## Model Details

- **Model:** Qwen/Qwen2.5-1.5B-Instruct
- **Size:** 1.5B parameters
- **Hardware:** CPU only (no `nvidia.com/gpu` request)
- **Framework:** vLLM (`vllm/vllm-openai-cpu` image), tool calling enabled via the `hermes` parser

## Deployment

Deployed via KServe InferenceService with:

- **Runtime:** vllm-cpu-runtime
- **minReplicas:** 1 (no GPU cost to offset, so it's kept warm to avoid CPU cold-start latency)

## Testing

```bash
curl http://qwen25-cpu-predictor.physical-ai-models.svc.cluster.local/v1/models

curl http://qwen25-cpu-predictor.physical-ai-models.svc.cluster.local/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen25-cpu","messages":[{"role":"user","content":"Hello!"}]}'
```
