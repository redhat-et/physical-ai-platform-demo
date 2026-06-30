# Adding Models to the Catalog

Each model lives in its own directory under `platform/base/models/<model-name>/` with a `kustomization.yaml` listing its resources.

**Two patterns exist:**

## Pattern A: KServe InferenceService (GPU models)

For models served by vLLM-Omni or another serving runtime. See `cosmos3-nano/` or `dreamzero/` as examples.

Required files:

- `kustomization.yaml` — lists `servingruntime.yaml` and `inferenceservice.yaml`
- `servingruntime.yaml` — defines the container image, args, ports, model format
- `inferenceservice.yaml` — defines the model URI, resource requests (CPU, memory, GPU), runtime reference

All resources deploy into namespace `physical-ai-models`.

## Pattern B: MaaS ExternalModel (external or mock endpoints)

For models hosted externally or via simple deployments. See `mocklm/` as an example.

Required files:

- `kustomization.yaml` — lists all resources
- `deployment.yaml` — the model server deployment (if self-hosted)
- `external-model.yaml` — `ExternalModel` CR registering the endpoint with MaaS
- `model-ref.yaml` — `MaaSModelRef` CR for catalog metadata
- `subscription.yaml` — `MaaSSubscription` CR defining access and rate limits
- `auth-policy.yaml` — `MaaSAuthPolicy` CR for authentication rules

## Wiring into an overlay

After adding the model directory, include it in the appropriate overlay's `kustomization.yaml`:

```yaml
resources:
  - ../../base
  - ../../base/models/<model-name>/
```
