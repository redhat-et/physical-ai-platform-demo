# Qwen3-Omni Model Deployment

Qwen3-Omni is an end-to-end omni-modal model from Alibaba Cloud, capable of understanding text, audio, images, and video, and generating text and speech.

## Model Details

- **Model:** Qwen/Qwen3-Omni-30B-A3B-Instruct
- **Type:** MoE (30B total, 3B active per token)
- **Size:** 65.7GB (BF16 on disk, loaded as FP8)
- **GPU:** 4x NVIDIA L4 (24GB each)
- **Framework:** vLLM Omni

## Deployment

Deployed via KServe InferenceService with:
- **Runtime:** vllm-omni-qwen3-runtime
- **Quantization:** FP8 (on-the-fly, ~33GB in GPU memory)
- **GPU Layout:** Thinker (TP=2) on GPUs 0,1 | Talker on GPU 2 | Code2Wav on GPU 3
- **Deploy Config:** Custom ConfigMap (`qwen3-omni-deploy-config`) with `max_model_len: 32768`
- **Stage Init Timeout:** 900s (model loading takes ~10 minutes)

## Llama Stack Patch

Qwen3-Omni defaults to `["text", "audio"]` output, which causes raw WAV audio data to appear in playground responses. The Llama Stack vLLM adapter is patched to inject `modalities: ["text"]` on all requests.

This patch is applied via the `LlamaStackDistribution` CR startup command:

```bash
oc patch llamastackdistribution lsd-genai-playground -n physical-ai-models --type=json -p '[
  {"op":"replace","path":"/spec/server/containerSpec/command","value":["/bin/sh","-c","python3 -c \"\np='"'"'/opt/app-root/lib/python3.12/site-packages/llama_stack/providers/remote/inference/vllm/vllm.py'"'"'\nt=open(p).read()\nif '"'"'modalities'"'"' not in t:\n  t=t.replace('"'"'return await super().openai_chat_completion(params)'"'"','"'"'if not params.model_extra.get(\\\"modalities\\\"):\\n            params.modalities = [\\\"text\\\"]\\n        return await super().openai_chat_completion(params)'"'"')\n  open(p,'"'"'w'"'"').write(t)\n\"\nllama stack run /etc/llama-stack/config.yaml"]}
]'
```

This is a cluster-level change (not in Git) because the BFF manages the LSD CR.

## Testing

```bash
# Health check
curl http://qwen3-omni-predictor.physical-ai-models.svc.cluster.local/v1/models

# Text chat (streaming)
curl http://qwen3-omni-predictor.physical-ai-models.svc.cluster.local/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-omni","messages":[{"role":"user","content":"Hello!"}],"modalities":["text"],"stream":true}'
```
