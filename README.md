# Physical AI Platform Demo

A proof-of-concept platform extending Red Hat OpenShift AI (RHOAI) to the
domain of physical AI — robotics, autonomous vehicles, digital twins.

Deploys the production RHOAI Models-as-a-Service (MaaS) stack with
Kuadrant/Red Hat Connectivity Link and registers physical AI models through
the governed gateway.

## Quickstart

```bash
# 1. Validate prerequisites
./platform/preflight.sh

# 2. Deploy (pick one profile)
oc apply -k platform/overlays/dev        # no GPU, mock models
oc apply -k platform/overlays/dev-gpu    # with GPU, real model serving
oc apply -k platform/overlays/demo       # full demo environment
```

See [platform/prerequisites.md](platform/prerequisites.md) for what must be
installed before deploying.

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

```
physical-ai-platform-demo/
├── platform/                  # deployable manifests
│   ├── prerequisites.md       # what must be installed first
│   ├── preflight.sh           # validates prerequisites
│   ├── base/                  # shared kustomize base
│   │   ├── maas/              # MaaS gateway, PostgreSQL
│   │   └── models/            # model registrations
│   │       └── mocklm/        # mock LM (OpenAI-compatible)
│   └── overlays/              # per-environment overlays
│       ├── dev/               # minimal, no GPU
│       ├── dev-gpu/           # GPU-enabled
│       └── demo/              # full demo
├── docs/                      # design specs and plans
└── notes/                     # gitignored: local snapshots, research
```

## Removing the Platform

```bash
oc delete -k platform/overlays/dev  # same overlay used to deploy
```
