# π0.5 LeRobot Checkpoint Deployment

This directory contains the Kubernetes resources required to deploy a **LeRobot-finetuned π0.5 checkpoint** on the Physical AI platform.

## Why this directory exists

The existing:

```text
platform/base/models/pi05/
```

deployment is intended for the standard OpenPI π0.5 serving path.

LeRobot fine-tuned checkpoints use a different checkpoint format and include their own configuration, preprocessing, normalization, and postprocessing state.

For that reason, this directory defines a separate deployment instead of modifying the existing `pi05` model resources.

## Serving backend

This model is served through:

```text
features/lerobot-server/
```

and uses the runtime image:

```text
quay.io/redhat-et/lerobot-server:latest
```

The checkpoint is therefore loaded natively through LeRobot rather than through the standard OpenPI policy loader.

## Checkpoint source

The default checkpoint configured in this directory is:

```text
execbat/pi05-robot-finetuned
```

The model download job downloads the complete Hugging Face repository into the model PVC.

This includes:

```text
config.json
model.safetensors
policy_preprocessor.json
policy_postprocessor.json
processor state files
```

The full checkpoint is mounted into the inference container at:

```text
/mnt/models
```

## Deployment flow

The deployment path is:

```text
Hugging Face checkpoint
        |
        v
model-download-job
        |
        v
pi05-checkpoint-model-cache PVC
        |
        v
InferenceService
        |
        v
lerobot-pi05-runtime
        |
        v
lerobot-server
        |
        v
LeRobot Pi05Policy
```

## Resources in this directory

The directory contains the Kubernetes resources required for the deployment, including:

```text
pvc.yaml
triton-cache-pvc.yaml
model-download-job.yaml
inferenceservice.yaml
httpscaledobject.yaml
kustomization.yaml
```

### `model-download-job.yaml`

Downloads the complete LeRobot checkpoint from Hugging Face.

### `pvc.yaml`

Stores the downloaded checkpoint.

### `triton-cache-pvc.yaml`

Stores compilation and runtime caches used by the inference server.

### Shared LeRobot runtime

The [shared LeRobot ServingRuntime](../../model-serving/lerobot/servingruntime.yaml)
defines how to launch `lerobot-server`.

It starts:

```text
quay.io/redhat-et/lerobot-server:latest
```

with the checkpoint mounted at:

```text
/mnt/models
```

### `inferenceservice.yaml`

Creates the KServe inference endpoint for the checkpoint.

The resulting service is exposed as:

```text
pi05-checkpoint-predictor
```

### `httpscaledobject.yaml`

Provides HTTP-based scaling behavior for the inference service.

## Relationship to the standard π0.5 deployment

### Opt-in LIBERO deployment config for vLLM-Omni

The [LIBERO deployment config](../pi05-libero/pi05-libero-deploy.yaml) describes
the LIBERO checkpoint's cameras, state, and actions. Use the same Omni runtime
for another supported π0.5 checkpoint with its own artifact and deploy config.
Do not reuse the LIBERO metadata preparation script for DROID.

Apply the [example InferenceService](../pi05-libero/inferenceservice.example.yaml)
after installing the shared `vllm-omni-runtime` and `pi05-libero-omni-config`.
The example selects one NVIDIA L4 and mounts 1 GiB of memory-backed storage
at `/dev/shm`. The default 64 MiB could not satisfy the runtime's 240 MiB
allocation at startup. KServe stages the pinned Hugging Face checkpoint at
`/mnt/models`; the `huggingface-token` Secret must grant access to the gated
`google/paligemma-3b-pt-224` tokenizer.

The default overlays do not include this endpoint. Delete the example
InferenceService when the session ends to release its GPU.

Use the OpenPI WebSocket at `/v1/realtime/robot/openpi`, not the legacy route at
`/`. Send the two LIBERO camera keys (`observation.images.image` and
`observation.images.image2`) and an 8-value `state`. The runtime preprocesses
256×256 camera frames to 224×224 and returns a 10×7 action array. This checkpoint
does not ship normalization statistics; inputs and outputs stay in model space.
Verify action agreement in simulation before using the endpoint for control.

Verify WebSocket upgrades through your client route and cold starts before
enabling scale-to-zero.

### Existing runtimes

The standard OpenPI deployment remains available under:

```text
platform/base/models/pi05/
```

This directory does not replace it.

The intended separation is:

```text
platform/base/models/pi05/
    -> openpi-server
    -> standard OpenPI π0.5 model

platform/base/models/pi05_checkpoint/
    -> lerobot-server
    -> LeRobot-finetuned π0.5 checkpoint
```

This allows both serving paths to coexist on the platform.

## Playground integration

The checkpoint can be exposed as a separate model option in the playground configuration.

For example:

```text
pi0.5
pi0.5 Checkpoint
```

The standard π0.5 model continues to use the OpenPI backend, while the checkpoint variant uses the LeRobot backend defined here.

## Summary

This directory exists to deploy LeRobot-finetuned π0.5 checkpoints without modifying the existing standard OpenPI serving path.

All checkpoint-specific loading is handled by `lerobot-server`.
