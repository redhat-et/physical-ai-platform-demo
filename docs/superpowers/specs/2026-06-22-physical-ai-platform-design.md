# Physical AI Platform Demo — Design Spec

## Context

This project is a proof-of-concept "Physical AI Platform" that extends Red Hat AI Enterprise (RHAIE) / OpenShift AI (RHOAI) to the domain of physical AI — robotics, autonomous vehicles, digital twins. The goal is to demonstrate that RHOAI's model serving, model-as-a-service (MaaS), and governance capabilities can be extended to serve physical AI models (e.g., Vision-Language-Action models) using the real product stack.

The PoC must be:

- **Reproducible** — any developer can clone the repo and deploy the platform on their own cluster
- **Built on the real product stack** — uses production RHOAI components (KServe, MaaS with Kuadrant/RHCL), not lab substitutes (LiteMaaS/LiteLLM)
- **Portable across environments** — the same manifests work on minimal development clusters and full demo environments, differing only in resource allocation and model backends

## Environment Profiles

Instead of targeting specific clusters (e.g., "my SNO" or "Red Hat Demo Platform"), the platform defines three profiles along the dimensions of resource availability and intended use:

| Profile | GPU | Models | Use Case |
|---------|-----|--------|----------|
| **dev** | No | ExternalModel (proxied) or mock servers | Local development, CI, testing the gateway/catalog layer |
| **dev-gpu** | Yes (1+) | Real model serving (vLLM + weights) | Development with actual inference |
| **demo** | Yes (multiple) | Full model set | Live demonstrations, workshops |

Key design principle: the architecture is identical across all profiles. The difference is only what model backends are registered (mock/external vs. real) and how much resource is allocated. "Production" is not a separate profile — it's `demo` with operational concerns (HA, monitoring, backup) that are outside the scope of this PoC.

## Repo Structure

```text
physical-ai-platform-demo/
├── .gitignore
├── README.md
├── platform/
│   ├── prerequisites.md            # documented prerequisites with versions
│   ├── preflight.sh                # validates prerequisites are met
│   ├── base/                       # shared kustomize base
│   │   ├── kustomization.yaml
│   │   ├── namespace.yaml          # platform namespace(s)
│   │   ├── dsc-patch.yaml          # DSC patch: enable MaaS, set Headed
│   │   ├── maas/                   # MaaS resources
│   │   │   ├── kustomization.yaml
│   │   │   ├── gateway.yaml        # maas-default-gateway
│   │   │   ├── tenant.yaml         # default-tenant Tenant CR
│   │   │   └── postgresql.yaml     # dev PostgreSQL (StatefulSet)
│   │   └── models/                 # model registrations
│   │       ├── kustomization.yaml
│   │       └── mocklm/             # mock LM model
│   │           ├── kustomization.yaml
│   │           ├── deployment.yaml          # or inferenceservice.yaml
│   │           ├── external-model.yaml      # ExternalModel CR
│   │           ├── model-ref.yaml           # MaaSModelRef
│   │           ├── subscription.yaml        # MaaSSubscription
│   │           └── auth-policy.yaml         # MaaSAuthPolicy
│   └── overlays/
│       ├── dev/                    # minimal, no GPU
│       │   ├── kustomization.yaml
│       │   └── resource-patches.yaml
│       ├── dev-gpu/                # with real model serving
│       │   ├── kustomization.yaml
│       │   ├── resource-patches.yaml
│       │   └── models/             # additional real models
│       └── demo/                   # full demo environment
│           ├── kustomization.yaml
│           ├── resource-patches.yaml
│           └── models/             # full model set
├── docs/                           # design specs, plans
└── notes/                          # gitignored: local snapshots, research, scripts
```

## Prerequisites

The platform does not install cluster-level operators. These are prerequisites that must be present before deploying:

| Prerequisite | Version | Required For | Notes |
|---|---|---|---|
| OpenShift Container Platform | 4.19.9+ | Gateway API CRDs | Native in 4.19+ |
| Red Hat OpenShift AI (RHOAI) | 3.4+ | KServe, MaaS, Model Registry | Operator from OperatorHub |
| Red Hat Connectivity Link (RHCL) | 1.3+ | Kuadrant, Authorino, Limitador | Included in RHAI SKU for MaaS |
| cert-manager | 1.x | TLS certificates | Operator from OperatorHub |
| Storage provisioner | any | PVC support | Any CSI provisioner with a default StorageClass |
| User Workload Monitoring | enabled | MaaS metrics | OCP built-in, needs enablement |

### Preflight Script

`platform/preflight.sh` validates that prerequisites are met. It **reports** what's missing but does **not** install anything — installation is environment-specific and should be done by a cluster admin.

Checks:

- `oc` is available and logged in with cluster-admin
- OCP version ≥ 4.19.9
- RHOAI operator installed and ≥ 3.4
- RHCL operator installed and ≥ 1.3
- cert-manager operator installed
- Default StorageClass exists
- User Workload Monitoring is enabled

Output: a pass/fail checklist with actionable messages for each failure.

## Deployment

### What the Platform Deploys

The platform manifests manage resources that sit *on top of* the prerequisite operators:

1. **DSC patch** — patches the existing DataScienceCluster to enable MaaS (`modelsAsService: Managed`, `rawDeploymentServiceConfig: Headed`)
2. **MaaS infrastructure** — Gateway, Tenant CR, dev PostgreSQL instance, TLS configuration
3. **Model registrations** — ExternalModel, MaaSModelRef, MaaSSubscription, MaaSAuthPolicy per model
4. **Model backends** — mock model deployments (dev profile) or InferenceService CRs (dev-gpu/demo profiles)

### Deployment Flow

```
# 1. Validate prerequisites
./platform/preflight.sh

# 2. Deploy the platform (pick one profile)
oc apply -k platform/overlays/dev        # minimal, no GPU
oc apply -k platform/overlays/dev-gpu    # with GPU model serving
oc apply -k platform/overlays/demo       # full demo
```

### Profile Differences

| Aspect | dev | dev-gpu | demo |
|--------|-----|---------|------|
| PostgreSQL | dev instance (StatefulSet) | dev instance | external (production) |
| Model backends | mocklm Deployment | vLLM InferenceService | multiple InferenceServices |
| MaaS gateway replicas | 1 | 1 | 2+ |
| Resource requests | minimal | moderate | full |
| ExternalModel CRs | mock server (in-cluster) | optional | optional |

## Models

### Mock Models (dev profile)

Uses [mock-models](https://github.com/fzdarsky/mock-models) — lightweight servers implementing standard APIs with canned responses:

- **mocklm** — OpenAI Chat Completions API, ~50m CPU / 64Mi memory
- **mockvla** (future) — VLA inference API (protocol TBD: OpenPI or LeRobot)

Deployed as plain Kubernetes Deployments, registered with MaaS via ExternalModel CRs pointing to the in-cluster Service.

### Real Models (dev-gpu / demo profiles)

Deployed as KServe InferenceService CRs with vLLM ServingRuntime. Registered with MaaS via MaaSModelRef pointing to the InferenceService.

## Design Decisions

- **Kustomize over Helm** — simpler for a PoC, no templating engine needed, native `oc apply -k` support. Overlays handle per-environment differences cleanly.
- **Prerequisites are documented, not automated** — operator installation varies by environment (OperatorHub channel, approval strategy, namespace conventions). A preflight check is more robust than a fragile installer script.
- **Environment profiles by capability, not by cluster name** — `dev`/`dev-gpu`/`demo` instead of "SNO"/"Demo Platform". Any cluster that meets the prerequisites can run any profile.
- **Real product stack only** — no LiteMaaS, no LiteLLM. The MaaS layer uses Kuadrant/RHCL + Authorino + Gateway API, which is the GA product in RHOAI 3.4.
- **Dev PostgreSQL is intentionally simple** — a single-replica StatefulSet with hostpath storage. Not production-grade, but sufficient for development and demos.
- **Mock models live in a separate repo** — [mock-models](https://github.com/fzdarsky/mock-models) is general-purpose and not specific to this platform. The platform repo references the container images, not the source.
