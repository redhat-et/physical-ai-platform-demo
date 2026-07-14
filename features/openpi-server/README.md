# OpenPI Server

Container image for serving pi0.5 and other OpenPI-compatible robot policies via KServe on OpenShift.

## Upstream

Built from [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi).

## Image

Published to `quay.io/redhat-et/openpi-server:latest`.

## Build

```bash
make build                          # build with default (main branch)
make build OPENPI_REF=v0.2.0       # pin to a specific tag or commit
make push                          # push to quay.io (requires login)
make deploy                        # build + push
```

The build clones the openpi repo, installs dependencies via `uv`, and patches
the bundled transformers library. Expect ~10-15 min build time and an ~8-10 GB image
(CUDA runtime + ML dependencies).

## Usage

The image is referenced by the `openpi-runtime` ServingRuntime in
`platform/base/models/pi05/servingruntime.yaml`. The policy config and model
directory are passed via the ServingRuntime's container `command`/`args`.
