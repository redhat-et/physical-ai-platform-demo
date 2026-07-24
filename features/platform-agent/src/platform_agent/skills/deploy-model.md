---
name: deploy-model
description: Use before sizing hardware or drafting manifests for a model that fits an existing runtime pattern (stock vLLM chat, or vLLM-Omni multimodal). For a model needing a runtime this platform hasn't run before, use new-model-runtime instead.
---
DEPLOY MODEL — for hardware sizing or adding a new model to the catalog that
fits an existing runtime pattern, in order:

1. Call list_cluster_gpus to see real GPU capacity — never guess it.
2. Call estimate_model_footprint with the target Hugging Face repo id to get a real recommended tensor_parallel_size — never guess that either.
3. Only then call generate_model_manifests using the values from steps 1-2. generate_model_manifests only knows two runtime templates: stock vLLM (output_kind='chat') and vLLM-Omni (output_kind='image'/'video'). If the target model's real serving mechanism is neither of those — e.g. it needs its own native server/CLI — stop here and use the new-model-runtime skill instead; forcing an unfamiliar runtime through these templates produces a broken deployment.
4. Return the generated YAML to the user verbatim in fenced code blocks — do not paraphrase, shorten, or summarize it.
5. Tell the user this is a draft only: this platform uses GitOps (ArgoCD self-heal + prune), so nothing is actually deployed until a human saves the files, wires them into an overlay, and merges a PR. You cannot deploy a model yourself.
