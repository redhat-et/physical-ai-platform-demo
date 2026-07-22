---
name: new-model-runtime
description: Use when onboarding a model whose serving runtime this platform hasn't run before (not stock vLLM chat or vLLM-Omni) — its own native server/CLI, not something generate_model_manifests can template. See also the deploy-model skill for models that DO fit an existing runtime.
---
NEW MODEL RUNTIME — for a model that doesn't fit generate_model_manifests'
two templates (stock vLLM for chat, vLLM-Omni for image/video). Hand-draft
the manifests instead; don't force it through generate_model_manifests.
Example: pi0.5 doesn't run under vLLM-Omni at all (unsupported upstream —
vllm-project/vllm-omni#4136) — it runs via Physical Intelligence's own
`openpi` server. Treat any unfamiliar serving mechanism as this case by
default.

STEP 0 — GET THE REAL SERVING DETAILS, DON'T GUESS: container image, real
entrypoint/command/args, port, required env vars, API shape (an existing
output_kind, or none). If you don't have these, ask for them or a link to
the model's serving docs — inventing a CLI flag or entrypoint produces a
broken pod, unlike a wrong hardware estimate which fails safely. Still call
list_cluster_gpus/estimate_model_footprint for a starting point, but treat
the footprint estimate as weaker than usual here — it's sized from raw
parameter bytes, and an unfamiliar runtime's real memory/compute behavior
(compiled-kernel caches, warmup, its own batching) may not track that.

FILE STRUCTURE — see docs/adding-models.md for the canonical two patterns
(KServe InferenceService for self-hosted models, MaaS ExternalModel for
externally-hosted ones); in practice almost every model needs both: a real
InferenceService AND the MaaS catalog-registration files
(external-model.yaml, model-ref.yaml, subscription.yaml, auth-policy.yaml —
these four are the same shape regardless of runtime, safe to copy the
structure from any existing model and rename). For the InferenceService
side, produce:

- `kustomization.yaml` — lists every file below.
- `servingruntime.yaml` — the REAL container image, command/args, ports,
  and env for this specific runtime. Do not reuse vLLM's `--served-model-name`/
  `--tensor-parallel-size` style args on a non-vLLM server; every runtime has
  its own CLI. Worked example, pi0.5's actual `openpi-runtime` (condensed):

  ```yaml
  apiVersion: serving.kserve.io/v1alpha1
  kind: ServingRuntime
  metadata:
    name: openpi-runtime
    namespace: physical-ai-models
  spec:
    containers:
    - name: kserve-container
      image: quay.io/redhat-et/openpi-server:latest
      imagePullPolicy: Always
      ports:
      - containerPort: 8000
      command: ["uv", "run"]
      args: ["serve_with_warmup.py", "policy:checkpoint", "--policy.config=pi05_droid", "--policy.dir=/mnt/models"]
      env:
      - name: TRITON_CACHE_DIR
        value: /cache/triton
      volumeMounts:
      - name: triton-cache
        mountPath: /cache
      resources:
        requests: {cpu: "2", memory: 24Gi, nvidia.com/gpu: "1"}
        limits: {cpu: "4", memory: 48Gi, nvidia.com/gpu: "1"}
    supportedModelFormats:
    - name: pytorch
      version: "2"
      autoSelect: true
  ```

- `inferenceservice.yaml` — `minReplicas: 0`, GPU `nodeSelector`, references
  the runtime above by name, mounts any extra volumes it needs (see the
  compiled-kernel-cache note below). Set
  `annotations: {physical-ai.io/output-kind: ...}` to a real output_kind
  (chat/image/video) ONLY if the runtime actually speaks one of call_model's
  supported API shapes — otherwise set it to `unsupported`, exactly like
  pi0.5's own InferenceService does, rather than inventing a mapping.
- `pvc.yaml` + `model-download-job.yaml` — model weights cache, same pattern
  as any other model, unless the runtime handles its own download.
- **Compiled-kernel-cache PVC, when relevant**: if the runtime does
  JIT/AOT compilation (Triton, `torch.compile`, similar), add a small
  dedicated PVC for that cache directory (see pi0.5's `triton-cache-pvc.yaml`,
  1Gi) and mount it — otherwise every scale-to-zero restart repays the full
  compile/warmup cost instead of reusing it.
- `httpscaledobject.yaml` — only if HTTP-triggered scale-from-zero applies;
  skip it if the runtime can't be scaled that way.
- `metrics-service.yaml` + `prometheus-rule.yaml` — optional, copy from an
  existing model if the runtime exposes Prometheus metrics.
- `README.md` — document the model and, specifically, WHY it needs a custom
  runtime instead of vLLM/vLLM-Omni (see pi0.5's README: one paragraph on
  what it is, one on why vLLM-Omni doesn't work for it, with a link to the
  upstream issue/repo).

Return every generated file to the user verbatim in fenced code blocks, one
per file — do not paraphrase or summarize. Same GitOps caveat as
deploy-model: this is a draft only, nothing is deployed until a human saves
the files, wires the model directory into an overlay's kustomization.yaml,
and merges a PR — you cannot deploy a model yourself.
