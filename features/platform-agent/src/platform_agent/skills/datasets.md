---
name: datasets
description: Use before searching for, inspecting, validating, or pulling a dataset for fine-tuning or general use, or when asked what data a robot-policy model (pi0.5, DreamZero) was trained or fine-tuned on.
---
DATASETS — in order:

1. list_staged_datasets first — don't re-pull an already-staged dataset.
2. For a named robot-policy model, search using the facts in APPLYING THE CHECKLIST below, not the model's own name. 'pi05' as a query returns any embodiment anyone used with it; 'droid' + expected_robot_type='franka' returns what the recipe actually needs.
3. get_dataset_info for size, license, gated status, and schema. See DATASET COMPATIBILITY CHECKLIST — this is one dimension, not the whole picture.
4. validate_lerobot_dataset(dataset_repo_id=..., expected_exterior_cameras=..., expected_wrist_cameras=..., expected_action_dim=...) using the values in APPLYING THE CHECKLIST — read them fresh each time, don't rely on memory. Never invent expected_feature_keys; omit it and read the schema yourself.
5. EXCEPTION TO RULE 1: never call pull_dataset same-turn as get_dataset_info. Show size/license/gated status, get explicit go-ahead first.
6. After pulling, get_dataset_job_status to confirm success before saying the dataset is ready.

DATASET COMPATIBILITY CHECKLIST — use for any robot-policy dataset + model
pairing on this platform, not just pi0.5/DROID. Weight dimensions by how
adaptable the model is: a model pretrained across many embodiments
tolerates deviation in Embodiment, Perceptual Setup, and Environment
Diversity. A model fine-tuned on one narrow dataset — like this platform's
pi0.5 recipe — can't lean on that, so nearly every dimension matters.
Action Space & Representation and Format & Tooling Compatibility are
critical regardless of adaptability: a mismatch there corrupts the
training signal itself, not just downstream generalization.

| # | Dimension | Checklist parameters | How to check it here |
|---|---|---|---|
| 1 | Embodiment & Kinematics | Arm/platform, DoF, kinematic chain, gripper type, gripper encoding + polarity, joint/workspace limits | `robot_type` from validate_lerobot_dataset (often unpopulated) + the Features list. DoF/kinematics/limits: no tool exposes these — use the platform's published specs or ask the user. |
| 2 | Action Space & Representation | Joint vs. cartesian, absolute vs. delta, coordinate frame origin, action dimensionality, chunk horizon | validate_lerobot_dataset reports action shape. Encoding and frame aren't in the schema — use get_dataset_rows to compare `action` against a labeled `observation.state.*` field at adjacent row indices. |
| 3 | Perceptual Setup | Camera count/mounting, extrinsics/intrinsics, FOV, resolution, stereo/mono, depth/tactile | search_compatible_lerobot_datasets/validate_lerobot_dataset count camera features by name substring; search_datasets's `tags` param can pre-filter by `modality:video`/`modality:image` at search time. For calibration/FOV/depth/tactile, get_dataset_file('README.md') — not always documented, but worth checking before assuming. |
| 4 | Dynamics & Control Quality | Recording frequency, control latency, motion smoothness, teleop noise, idle-frame density | `fps` from validate_lerobot_dataset. Latency/smoothness/teleop noise: not exposed — check documented collection methodology (get_dataset_file('README.md')). |
| 5 | Normalization & Statistics | Precomputed stats availability, normalization scheme the recipe needs | get_dataset_file('meta/stats.json') to check for precomputed mean/std or q01/q99 — confirm the target recipe actually needs it before treating absence as disqualifying. |
| 6 | Format & Tooling Compatibility | Storage format, schema/codebase version, feature key naming, metadata completeness | validate_lerobot_dataset checks `codebase_version` and returns the raw Features list — exact key names and gaps are visible there. search_datasets's `tags` param can pre-filter by `format:parquet`/`LeRobot` at search time. |
| 7 | Task Structure & Annotations | Episode/trajectory length, success/failure labeling, instruction presence + specificity, task metadata | `total_episodes` from validate_lerobot_dataset; get_dataset_rows across a few indices shows whether a task/language field is populated and how specific it actually is. |
| 8 | Scale & Composition | Total episodes/frames/hours, minimum viable episode count, storage size, success/fail composition | get_dataset_info reports GB; validate_lerobot_dataset reports `total_episodes` — check both against what the recipe needs (below). search_datasets's `size_category` pre-filters by row/frame-count bucket at search time — NOT a GB or episode-count proxy, use max_size_gb (search_compatible_lerobot_datasets) for an actual GB limit. |
| 9 | Environment & Task Diversity | Scene/task/object count, visual clutter, lighting, domain randomization | Not in the schema — get_dataset_file('README.md') for the dataset's own collection-methodology description. |
| 10 | Provenance, Identity & Licensing | Exact source/lineage, naming-confusion risk, license | get_dataset_info now also reports `Gated`/`Tags`/`Created`/`Last modified`; cross-check episode/frame counts against what a repo's name implies. search_datasets/search_compatible_lerobot_datasets can filter by `license`/`gated` at search time instead of checking each candidate after the fact. |

APPLYING THE CHECKLIST — THIS PLATFORM'S PI0.5 RECIPE (DROID, Distributed
Robot Interaction Dataset, droid-dataset.github.io / RSS 2024):

1. **Embodiment**: Franka Panda 7DoF arm, Robotiq 2F-85 parallel-jaw
   gripper.
2. **Action space**: 7-dim (matches expected_action_dim), but this is the
   dimension that actually needs checking, not just matching a count.
   DROID's native schema stores joint velocity, joint position, and
   cartesian position/velocity side by side; different rehosts expose
   different ones as the flat `action` feature, and the coordinate frame
   (base, wrist, camera) is rarely documented at all. `lerobot-train`/
   pi0.5 trains on whatever `action` contains without erroring — a
   dimension match means training will run, not that the output means
   what you'd assume. For any candidate, use get_dataset_rows to compare
   `action` against a labeled `observation.state.*` field at adjacent row
   indices before asserting compatibility — don't infer a rehost's
   encoding from another rehost's. Chunk horizon: this recipe doesn't
   override `chunk_size` from `lerobot/pi05_base`'s default of 50.
   openpi's own official DROID recipes use 15-16 instead, tuned to DROID's
   control frequency — this platform hasn't verified whether training at
   50 on DROID data is fine or should be overridden down to match.
3. **Perceptual setup**: 2 exterior Zed 2 stereo cameras + 1 wrist Zed
   Mini stereo camera. DROID ships depth + full camera calibration
   generally, but LeRobot conversions often drop depth — check a
   candidate's own Features list (or get_dataset_file('README.md')),
   don't assume it carries over.
4. **Control frequency**: 15fps (DROID and its rehosts). Check `fps`
   before assuming a candidate is a drop-in match — a different frequency
   changes the real-world horizon an action chunk covers. DROID's
   VR-teleop collection leaves many idle frames; this platform's tools
   don't filter them, so episode count alone overstates useful signal.
5. **Normalization**: this recipe sets
   `--policy.normalization_mapping='{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}'`
   instead of pi0.5's stock QUANTILES default. A candidate missing
   precomputed q01/q99 stats is not disqualified here.
6. **Format**: requires LeRobot v3.0, not v2.x — convert with
   `python -m lerobot.scripts.convert_dataset_v21_to_v30` or pin an
   older lerobot package. (Confirmed live against `huggingface/lerobot-gpu:latest`:
   `lerobot.datasets.v30.convert_dataset_v21_to_v30` no longer exists in that
   image -- the module moved to `lerobot.scripts`.) Feature key spelling varies across rehosts (e.g.
   `observation.image.X` vs `observation.images.X`) — read the actual
   Features list, don't assume a name.
7. **Annotations**: pi0.5 is language-conditioned — an episode needs a
   real task description, not just a populated field. DROID's official
   annotations cover ~95% of successful episodes, not the ~16,000
   unsuccessful ones.
8. **Scale**: 76,000 successful trajectories + ~16,000 unsuccessful (still
   released), 350 hours. This recipe reserves up to 5 episodes for eval
   and needs ≥1 left for training, so ≥2 episodes to run at all, and more
   than that to mean anything. Never infer episode count from a repo's
   name — names round, guess, or refer to a different subset than what's
   actually there; always read `total_episodes` from validate_lerobot_dataset.
9. **Diversity**: 564 scenes, 86 tasks, standardized hardware across 13
   institutions, collection spanning 52 buildings/3 continents (different
   axes, both real). Scenes are re-randomized roughly every 20 minutes
   (relighting, camera moves, object changes).
10. **Provenance/license**: see the naming caveat in #8. get_dataset_info
    reports license per candidate — don't assume it matches another DROID
    rehost.

**DreamZero** (GEAR-Dreams/DreamZero-DROID, inference-only, no fine-tuning
recipe): trained on `GEAR-Dreams/DreamZero-DROID-Data` — 57,774 episodes
(~131-145GB, 14.7M frames), a filtered DROID derivative (idle frames,
non-annotated and unsuccessful episodes removed). Not the "~75k" figure
sometimes quoted — that's DROID's overall annotation coverage, not this
repo's size. Uses relative joint positions as its action space, unlike the
cartesian encoding the raw LeRobot DROID ports use. `DreamZero-AgiBot` is a
separate checkpoint on different data (AgiBot G1 teleop) — don't conflate
it with DreamZero-DROID. Answer training-data questions from this; don't
offer to fine-tune it — no recipe exists.

Don't recommend a fixed list of "known good" repo ids — Hub datasets get
renamed, replaced, and superseded. Use search_compatible_lerobot_datasets
to find current candidates, and size them against what the user actually
wants: this recipe's dry-run default (~50 steps) only needs pipeline-scale
data (order of 100 episodes) to validate end-to-end, not to produce a
usable policy — a real production fine-tune needs the full checklist
applied at a much larger scale, and the storage/GPU-time that implies.

BROADER LANDSCAPE (if asked about a model/dataset not on this platform):
DROID is one of ~60 datasets in Open X-Embodiment (OXE), alongside
Bridge-v2, RT-1, and others. Format/schema compatibility doesn't guarantee
training-compatibility — OpenVLA dropped DROID from its own OXE mixture
partway through training because it hurt accuracy, despite matching
format. Don't assert a dataset for a model not covered here without
checking its actual documentation first.
