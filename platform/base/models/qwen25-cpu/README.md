# Qwen2.5-1.5B-Instruct (CPU)

A small instruction-tuned LLM served on CPU-only nodes via vLLM's CPU backend, for
environments/experiments where GPU capacity isn't available or needed.

## Model Details

- **Model:** Qwen/Qwen2.5-1.5B-Instruct
- **Size:** 1.5B parameters
- **Hardware:** CPU only (no `nvidia.com/gpu` request)
- **Runtime:** RHOAI `vllm-cpu-x86-runtime`, with the `hermes` tool parser

## Deployment

The `dev-gpu` overlay installs the shared runtime but does not start this
model. After pruning any previous GitOps deployment with this name, start
one CPU replica from the repository root:

```bash
oc create -k platform/base/models/qwen25-cpu
```

KServe stages the pinned Hugging Face checkpoint in each new pod. There is
no MaaS registration or request-driven autoscaler. Remove the endpoint when
done with `oc delete -k platform/base/models/qwen25-cpu`.

## Testing

```bash
curl http://qwen25-cpu-predictor.physical-ai-models.svc.cluster.local/v1/models

curl http://qwen25-cpu-predictor.physical-ai-models.svc.cluster.local/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen25-cpu","messages":[{"role":"user","content":"Hello!"}]}'
```
