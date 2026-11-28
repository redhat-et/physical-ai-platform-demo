# Qwen3-Omni Model Deployment

Qwen3-Omni is a multimodal model that accepts text, audio, images, and video.
This example uses `Qwen/Qwen3-Omni-30B-A3B-Instruct`.

## Deployment

The GPU overlays install the shared `vllm-omni-runtime` but do not start this
model. After pruning any previous GitOps deployment with this name, start one
replica from the repository root:

```bash
oc create -k platform/base/models/qwen3-omni
```

The manifest requests one NVIDIA L40S (48 GB) and mounts
`qwen3-omni-deploy-config` for FP8 and a 32K context. KServe stages the pinned
Hugging Face checkpoint in each new pod. There is no MaaS registration or
request-driven autoscaler. Release the GPU when done with
`oc delete -k platform/base/models/qwen3-omni`.

## Testing

From inside the cluster, check the model and make a text-only request:

```bash
curl http://qwen3-omni-predictor.physical-ai-models.svc.cluster.local/v1/models

curl http://qwen3-omni-predictor.physical-ai-models.svc.cluster.local/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-omni","messages":[{"role":"user","content":"Hello!"}],"modalities":["text"],"stream":true}'
```
