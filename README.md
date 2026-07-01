# Physical AI Platform Demo

A proof-of-concept platform extending Red Hat OpenShift AI (RHOAI) to
physical AI, focusing on video analytics and robotics use cases.

> [!WARNING]
> This project is experimental and under active development. Expect breaking
> changes.

Deploys the RHOAI Models-as-a-Service (MaaS) stack with Red Hat
Connectivity Link, providing a model catalog where users request an
endpoint and KServe handles scale-to-zero / scale-from-zero on demand.
Models are served through vLLM-Omni or partner runtimes.

## Quickstart

See [platform/prerequisites.md](platform/prerequisites.md) for operator and
cluster requirements. Run `./platform/preflight.sh` to validate. Use
`--generate-secrets` to create required database secrets with random passwords:

```bash
./platform/preflight.sh                # check prerequisites
./platform/preflight.sh --generate-secrets  # create database secrets if missing
```

### GitOps (recommended)

Apply the ArgoCD Application for your chosen profile — see
[argocd/README.md](argocd/README.md) for details:

```bash
oc apply -f argocd/application-dev-gpu.yaml
oc apply -f argocd/permissions.yaml
```

### Manual

```bash
oc apply -k platform/overlays/dev        # no GPU, mock models
oc apply -k platform/overlays/dev-gpu    # with GPU, real model serving
oc apply -k platform/overlays/demo       # full demo environment
```

## Environment Profiles

| Profile   | GPU | Models                         | Use Case                    |
|-----------|-----|--------------------------------|-----------------------------|
| dev       | No  | Mock servers / external proxy  | Development, CI, testing    |
| dev-gpu   | Yes | Real model serving (vLLM)      | Development with inference  |
| demo      | Yes | Full model set                 | Demonstrations, workshops   |

The architecture is identical across profiles. The difference is what model
backends are registered (mock vs. real) and resource allocation. Any cluster
meeting the prerequisites can run any profile.

## Repository Layout

```text
physical-ai-platform-demo/
├── platform/                  # deployable manifests
│   ├── prerequisites.md       # what must be installed first
│   ├── preflight.sh           # validates prerequisites
│   ├── base/                  # shared kustomize base
│   │   ├── namespace.yaml     # physical-ai, physical-ai-models, models-as-a-service
│   │   ├── dsc-patch.yaml     # DataScienceCluster (enables KServe + MaaS)
│   │   ├── maas/              # MaaS gateway, tenant, PostgreSQL
│   │   └── models/            # model catalog (each model = subdirectory)
│   │       ├── mocklm/        # mock LM (OpenAI-compatible, no GPU)
│   │       ├── cosmos3-nano/  # NVIDIA Cosmos 3 Nano world model
│   │       └── dreamzero/     # DreamZero robot policy model
│   └── overlays/              # per-environment overlays
│       ├── dev/               # minimal, no GPU
│       ├── dev-gpu/           # GPU-enabled development
│       └── demo/              # full demo with all models
├── argocd/                    # ArgoCD Application CRs and RBAC
├── docs/                      # guides (e.g. adding-models.md)
└── notes/                     # gitignored: local snapshots, research
```

To add models to the catalog, see [docs/adding-models.md](docs/adding-models.md).

## Development

### Prerequisites

- [uv](https://docs.astral.sh/uv/) — runs Python tools (yamllint) without a venv
- [kustomize](https://kubectl.docs.kubernetes.io/installation/kustomize/) — builds overlay manifests
- [kubeconform](https://github.com/yannh/kubeconform) — validates manifests against K8s schemas

On macOS: `brew install uv kustomize kubeconform`

### Validation

Run all checks (YAML lint, kustomize build, kubeconform):

```bash
make validate
```

Or run individual targets:

```bash
make lint              # YAML syntax and style
make kustomize-build   # build all overlays
make kubeconform       # schema validation
```

CI runs `make validate` on every PR that touches `platform/`.

## Removing the Platform

```bash
oc delete -k platform/overlays/dev  # same overlay used to deploy
```
