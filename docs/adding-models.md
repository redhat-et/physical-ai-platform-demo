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

## README

Every model directory includes a `README.md` with at minimum:

- `## Model Details` — HF repo, license, size, GPU.
- Its serving/deployment section (e.g. `## Serving Runtime` for a custom
  runtime, `## Deployment` otherwise).

A robot-policy model with a real dataset-compatibility question — a
fine-tuning recipe, or documented training data worth explaining — also
gets a `## Dataset Compatibility` section. Use that exact heading text: the
platform agent's `get_model_readme(model_name, section='Dataset Compatibility')`
tool matches on it verbatim, and a typo'd heading just means the agent falls
back to "no such section" instead of finding it. Don't add this section to
a model with nothing to check a candidate dataset against (a stock chat/LLM
model, a vision-only world model with no fine-tuning path) — the agent's
tool already handles a missing section gracefully.

When a real fine-tuning recipe exists, format the section as a table so it
lines up with the `datasets` skill's own `DATASET COMPATIBILITY CHECKLIST`
table row for row:

| # | Dimension | Priority | This model's specifics |
|---|---|---|---|
| 1 | Embodiment & Kinematics | ... | ... |

Use the exact same `#`/Dimension names and order as the skill's table
(Embodiment & Kinematics, Action Space & Representation, Perceptual Setup,
Dynamics & Control Quality, Normalization & Statistics, Format & Tooling
Compatibility, Task Structure & Annotations, Scale & Composition,
Environment & Task Diversity, Provenance/Identity & Licensing), so an agent
reading both tables can match rows directly instead of re-deriving the
correspondence. Priority is one of Critical / Adjustable / Minor, judged
specifically for that model + recipe — see `platform/base/models/pi05/
README.md` for the reasoning pattern (which facts are hard requirements vs.
recipe flags vs. cosmetic) — not copied from another model's table.

A model with no real fine-tuning recipe (inference-only, or no dataset
compatibility question to check) can use plain prose instead of the table —
see `platform/base/models/dreamzero/README.md`, which documents training
data provenance without a checklist to evaluate candidates against.

## Wiring into an overlay

After adding the model directory, include it in the appropriate overlay's `kustomization.yaml`:

```yaml
resources:
  - ../../base
  - ../../base/models/<model-name>/
```
