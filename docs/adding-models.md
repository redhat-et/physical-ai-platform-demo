# Adding Models to the Catalog

## AI hub model cards

To add a public, non-gated Hugging Face model to **AI hub → Models → Catalog → Physical AI models**,
add its full `owner/repository` ID to `includedModels` on the `physical_ai_hf` source in
[`platform/base/model-catalog/sources.yaml`](../platform/base/model-catalog/sources.yaml).
Use exact IDs and leave `allowedOrganization` unset so the source can span several organizations.
The static [MockLM cards](../platform/base/model-catalog/catalog.yaml) share the `Physical AI` label.

RHOAI reads the Hub repository's `main` README and metadata and supplies an `hf://` model location
for Register and Deploy. The dashboard title and provider come from the Hub, with no per-model
overrides or revision pin in the native source. Add or update card content in the Hub repository,
not in the static catalog. A catalog entry does not configure a serving runtime.

After syncing `model-catalog-sources`, check the catalog deployment is ready and the UI lists all
expected entries with their cards. Verify that Register and Deploy open on the imported models;
an Argo CD sync alone does not verify that RHOAI loaded the source.

## Serving manifests

Each model lives in its own directory under `platform/base/models/<model-name>/` with a `kustomization.yaml` listing its resources.

**Two patterns exist:**

## Pattern A: KServe InferenceService (GPU models)

See `cosmos3-nano/` and `dreamzero/` for examples. Both select the shared vLLM-Omni runtime under [`platform/base/model-serving/`](../platform/base/model-serving/). Each overlay includes the runtime definitions it needs.

Required files:

- `kustomization.yaml` — lists the model's resources
- `inferenceservice.yaml` — selects the runtime and defines the model URI, launch arguments, and resource requests

Reuse a compatible `ServingRuntime` rather than defining one per checkpoint. For
UI-deployed fine-tunes, register the complete serving artifact as a model
version and deploy it under a new name. Use a pinned Hugging Face revision
(`hf://`), immutable S3 prefix (`s3://`), or OCI ModelCar by digest (`oci://`).
Google Cloud Storage (`gs://`) is another supported KServe storage source;
check credentials and access in the deployment project. The runtime consumes
the staged files at `/mnt/models` regardless of the source.

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
fine-tuning recipe, or documented training data worth explaining — does
NOT get this in its own README anymore. That content now lives in the
platform agent's skills repo, under the `model-specs` skill's
`references/<model_name>.md` (e.g. `references/pi05.md`), fetched via the
`get_model_reference(model_name)` tool. Add or update a model's dataset
compatibility spec there instead — this repo's model directories only
cover Model Details / Serving / Deployment.

When a real fine-tuning recipe exists, format the reference as a table so
it lines up with the `datasets` skill's own `DATASET COMPATIBILITY
CHECKLIST` table row for row:

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
specifically for that model + recipe — see the skills repo's
`references/pi05.md` for the reasoning pattern (which facts are hard
requirements vs. recipe flags vs. cosmetic) — not copied from another
model's table.

A model with no real fine-tuning recipe (inference-only, or no dataset
compatibility question to check) can use plain prose instead of the table —
see the skills repo's `references/dreamzero.md`, which documents training
data provenance without a checklist to evaluate candidates against.

## Wiring into an overlay

After adding the model directory, include it in the appropriate overlay's `kustomization.yaml`:

```yaml
resources:
  - ../../base
  - ../../base/model-serving/vllm-omni/  # once per overlay, if needed
  - ../../base/models/<model-name>/
```
