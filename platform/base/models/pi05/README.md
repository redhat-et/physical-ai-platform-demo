# pi0.5 — Physical Intelligence Robot Policy Model

pi0.5 is a 4B-parameter Vision-Language-Action (VLA) model from Physical Intelligence
that achieves 5-10 Hz inference on a single L40S GPU.

## Model Details

- **HuggingFace repo**: [lerobot/pi05_base](https://huggingface.co/lerobot/pi05_base)
- **License**: Gemma (see HuggingFace model card)
- **Size**: ~16 GB
- **GPU**: 1x NVIDIA L40S

## Serving Runtime

pi0.5 uses the **native OpenPI server** from Physical Intelligence's
[openpi](https://github.com/Physical-Intelligence/openpi) repository, NOT vLLM-Omni.
vLLM-Omni does not support pi0.5 (see vllm-project/vllm-omni#4136).

The `openpi-runtime` ServingRuntime runs the OpenPI policy server with the `pi05_droid`
configuration.
