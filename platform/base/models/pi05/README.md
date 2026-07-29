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

## Dataset Compatibility

This platform's fine-tuning recipe trains `lerobot/pi05_base` via
`lerobot-train`. The recipe is dataset-agnostic — any LeRobot v3.0 dataset
works — so the checklist below describes what pi0.5/this recipe itself
requires or adjusts for, not the properties of any one dataset. `#`/
Dimension match the datasets skill's checklist table row for row; Priority
is specific to this model + recipe.

| # | Dimension | Priority | This recipe's specifics |
|---|---|---|---|
| 1 | Embodiment & Kinematics | Minor | No fixed embodiment requirement — dimension mismatches auto-pad (see row 2). Closer embodiment/camera-placement similarity to whatever data this recipe was validated against improves transfer quality, not whether training runs. |
| 2 | Action Space & Representation | Critical | pi0.5 predicts absolute actions by default (recipe doesn't set `--policy.use_relative_actions`) — a delta/velocity-encoded `action` field is a real mismatch, not just style. Dimension *count* auto-pads to 32, so exact dimensionality isn't required. `chunk_size` defaults to 50 — should match your dataset's real control frequency, not be left at default blindly. |
| 3 | Perceptual Setup | Adjustable | No required camera count — fewer views pad via `--policy.empty_cameras`. Camera key names don't need to match the pretrained checkpoint's own naming (renaming actively breaks training). |
| 4 | Dynamics & Control Quality | Adjustable | Control frequency is handled via `--policy.chunk_size`, not a hard requirement — pick a value matching your dataset's fps (see row 2). Idle/no-op frames aren't filtered by this platform's tooling. |
| 5 | Normalization & Statistics | Minor | Recipe overrides to MEAN_STD (computed from your dataset), not QUANTILES — missing precomputed q01/q99 stats isn't disqualifying. |
| 6 | Format & Tooling Compatibility | Critical (version only) | Requires LeRobot v3.0 (`convert_dataset_to_v3` for v2.x). Feature key *spelling* doesn't matter for training itself. |
| 7 | Task Structure & Annotations | Critical | Needs a real, specific task description per episode (language-conditioned). `observation.state` must be meaningfully populated too — it's discretized into the language prompt. |
| 8 | Scale & Composition | Adjustable | No fixed minimum from the architecture itself. Real-world pi0.5 fine-tunes succeed with a few hundred episodes (~10 hrs); PI's own guidance is ~1-20 hrs for task adaptation. Needs ≥6 episodes total to run at all (5 reserved for eval). |
| 9 | Environment & Task Diversity | Minor | Diversity mostly matters for broad pretraining, not a narrow task-specific fine-tune — a single-task dataset is normal and expected here. |
| 10 | Provenance, Identity & Licensing | Minor | Doesn't affect whether training runs — check license/provenance per candidate rather than assuming from its name or apparent lineage. |
