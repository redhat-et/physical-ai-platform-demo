# AGENTS.md

## Project Overview

This repo deploys a **Physical AI Platform** on OpenShift, extending Red Hat OpenShift AI (RHOAI) to serve Physical AI models (world models, robot policies, video understanding) through the same governed infrastructure used for language models.

The platform provides a **model catalog** via MaaS (Models as a Service) where users order a model endpoint and KServe handles scale-to-zero / scale-from-zero on demand. Models are served primarily through vLLM-Omni (for Physical AI model types like diffusion transformers) or partner runtimes when simpler.

**Current focus**: expanding the model catalog for fast experimentation.
**Target use cases**: video analytics and robotics.

## Repository Layout

```text
platform/
  base/                     # Shared Kustomize base — always deployed
    namespace.yaml          # physical-ai, physical-ai-models, models-as-a-service
    dsc-patch.yaml          # DataScienceCluster config (KServe, MaaS, Dashboard)
    maas/                   # MaaS infrastructure (gateway, tenant, PostgreSQL)
    models/                 # Model catalog — each model is a subdirectory
      mocklm/               # Mock OpenAI-compatible server (no GPU, always on)
      cosmos3-nano/         # NVIDIA Cosmos 3 Nano world model (L40S GPU)
      dreamzero/            # DreamZero robot policy model (L40S GPU)
  overlays/
    dev/                    # No GPU models — CI and local testing
    dev-gpu/                # Optional GPU models (commented out by default)
    demo/                   # Full demo profile with GPU models enabled

argocd/
  application-{dev,dev-gpu,demo}.yaml # ArgoCD Application CRs
  permissions.yaml                    # RBAC for ArgoCD service account
```

## Deployment Model

- **GitOps via ArgoCD**: push to `main` → ArgoCD auto-syncs to cluster (self-heal + prune enabled)
- **Kustomize overlays**: overlays select which models from `base/models/` to include
- **Currently active**: `physical-ai-platform-dev-gpu` ArgoCD app tracking `platform/overlays/dev-gpu`
- **Cluster**: shared ROSA cluster (`api.emerg.pcbk.p1.openshiftapps.com:6443`)
- **Namespaces**: `physical-ai` (infra), `physical-ai-models` (model workloads), `models-as-a-service` (MaaS control plane)

## How to Add a Model to the Catalog

See [docs/adding-models.md](docs/adding-models.md) for the two patterns (KServe InferenceService for GPU models, MaaS ExternalModel for external endpoints) and the file structure to follow.

## Conventions

- **Manifests are plain Kustomize YAML** — no Helm charts, no templating beyond Kustomize
- **Models are swappable** — don't over-engineer model definitions; the platform layer matters, not specific models
- **Prefer vLLM-Omni** as serving runtime where it supports the model type; fall back to the simplest working runtime otherwise
- **Scale-to-zero**: InferenceServices should use `minReplicas: 0` where feasible to save GPU resources on the shared cluster
- **GPU node affinity**: GPU models use `nodeSelector` to target specific GPU types (e.g., `nvidia.com/gpu.product: NVIDIA-L40S`)

## Prerequisites

See `platform/prerequisites.md` for operator and cluster requirements:

- Red Hat OpenShift AI 3.4+
- Red Hat Connectivity Link 1.3+ (`rhcl-operator`, NOT the community `kuadrant-operator`)
- cert-manager Operator 1.x
- OpenShift 4.19.9+ (for native Gateway API CRDs)

Run `./platform/preflight.sh` to validate. Use `--generate-secrets` to create required database secrets (MaaS, MLflow) with random passwords if not already present.

## Cluster Access

Use `oc` (OpenShift CLI). Common commands:

```bash
oc get inferenceservices -n physical-ai-models    # list served models
oc get pods -n physical-ai-models                 # check model server pods
oc get externalmodels -n physical-ai-models       # list MaaS-registered models
oc logs -n physical-ai-models <pod>               # debug a model server
oc get applications -n openshift-gitops           # check ArgoCD sync status
```

## Validation

Run `make validate` before committing to catch YAML and Kustomize errors locally. It runs the same checks as CI:

```bash
make validate        # all checks: lint + kustomize-build + kubeconform
make lint            # yamllint only
make kustomize-build # build all overlays (dev, dev-gpu, demo)
make kubeconform     # schema-validate rendered manifests against K8s + CRD schemas
```

Requires `yamllint` (via `uvx`), `kustomize` (or `oc`), and `kubeconform`.

## Code Style

- Conventional Commits for commit messages (<https://www.conventionalcommits.org/>)
- YAML: 2-space indent
- Namespace always set explicitly in manifests — don't rely on Kustomize's namespace transformer
- **Always work on feature branches** — never push directly to `main`; all changes go through PRs

## What NOT to Do

- **Don't put strategy, plans, or competitive analysis in this repo** — it is public
- **Don't modify cluster-scoped resources** (operators, CRDs) from this repo — those are managed separately
- **Don't hardcode secrets** — use Secret references; the current PostgreSQL dev credentials are a known debt
