# Adding Models to the Catalog

## AI hub model cards

To add or refresh a public, non-gated Hugging Face model in **AI hub → Models → Catalog → Physical AI**,
edit [`hack/model-catalog.yaml`](../hack/model-catalog.yaml), then run from the repository root:

```bash
uv run --script hack/update-model-catalog.py
git diff -- platform/base/model-catalog/catalog.yaml
```

Keep `name` stable to preserve the card URL. `repo` is the Hugging Face `owner/repository` ID. The
generator follows `main` unless you specify a `revision` (branch, tag, or commit). Each output card
records the full commit SHA and links to the pinned Hugging Face snapshot and README. The tool fetches
metadata and the README, not the weights. It omits `artifacts` because the RHOAI 3.5 static YAML
importer rejects `hf://` URIs.

Use `overrides` for catalog fields such as `description`, `tasks`, and string-valued `customProperties`.
Use `supplement` for model requirements, links to serving guides, or reviewed evaluation summaries.
Keep demo-specific runtime flags, resource settings, and setup steps in the per-model serving docs.
Keep evaluation details tied to the revision tested; the generator does not import or verify EvalHub
reports. Review upstream license changes in the diff. Commit the allowlist and generated catalog together.

The generator leaves hand-maintained entries, including MockLM, alone. To remove a generated card,
remove it from the allowlist and rerun the tool. `--check` reports whether a refresh would change the
file without writing it (exit `1` for changes, `2` for errors).

## Serving manifests

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
  - ../../base/models/<model-name>/
```
