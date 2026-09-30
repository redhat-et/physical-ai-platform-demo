# LeRobot Server

This directory contains a dedicated inference server for running **LeRobot-finetuned π0.5 checkpoints**.

## Why this server exists

The platform already includes an `openpi-server` implementation for standard OpenPI checkpoints.

However, LeRobot fine-tuned π0.5 checkpoints are packaged differently. In addition to `model.safetensors`, they include their own:

- `config.json`
- policy preprocessor configuration
- policy postprocessor configuration
- normalization processor state
- action unnormalization state

Because of this, loading a LeRobot checkpoint through the existing OpenPI server would require manually reproducing the checkpoint configuration, preprocessing, normalization, and postprocessing inside OpenPI.

This server exists specifically to avoid that.

Instead of modifying the existing OpenPI inference path, `lerobot-server` loads the checkpoint using the standard LeRobot inference stack.

## Intended use

Use this server for checkpoints that were fine-tuned and saved with LeRobot, especially π0.5 checkpoints such as:

```text
execbat/pi05-robot-finetuned
```

The server loads:

```text
config.json
model.safetensors
policy_preprocessor.json
policy_postprocessor.json
processor *.safetensors files
```

directly from the checkpoint directory.

This ensures that inference uses the same configuration and preprocessing/postprocessing pipeline that was saved during training.

## Architecture

The inference path is:

```text
Playground
    |
    v
WebSocket protocol
    |
    v
lerobot-server
    |
    v
LeRobot checkpoint config
    |
    v
LeRobot preprocessor
    |
    v
Pi05Policy
    |
    v
LeRobot postprocessor
    |
    v
Action chunk
```

The OpenPI server remains unchanged and continues to serve standard OpenPI checkpoints.

## Relationship to `openpi-server`

The two servers serve different checkpoint formats:

```text
features/openpi-server/
    Standard OpenPI checkpoints

features/lerobot-server/
    LeRobot-finetuned π0.5 checkpoints
```

The goal is to keep these inference paths separate.

`lerobot-server` must not require changing the behavior of the existing OpenPI server.

## WebSocket compatibility

The server keeps the same WebSocket-style inference interface used by the playground.

Only the transport protocol is shared with the OpenPI ecosystem.

Model loading, preprocessing, inference, and postprocessing are handled by LeRobot.

## Container image

The server is built as:

```text
quay.io/redhat-et/lerobot-server:latest
```

Build locally with:

```bash
make build
```

Push with:

```bash
make push
```

Or build and push together:

```bash
make deploy
```

## Runtime

The Kubernetes runtime for this server is configured under:

```text
platform/base/models/pi05_checkpoint/
```

That model definition mounts the downloaded LeRobot checkpoint into:

```text
/mnt/models
```

and starts this server with:

```text
--checkpoint /mnt/models
```

## Summary

Use `lerobot-server` when the model checkpoint is a LeRobot π0.5 fine-tune.

Use the existing `openpi-server` for standard OpenPI checkpoints.

This separation keeps the standard OpenPI serving path untouched while allowing native LeRobot checkpoint inference.
