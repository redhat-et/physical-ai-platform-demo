---
name: manage-models
description: Use before scaling an already-deployed model up or down, or taking down a checkpoint deployment or a permanent catalog model.
---
MANAGE MODELS — day-2 operations on models that already exist. For adding a
brand-new catalog model, use deploy-model or new-model-runtime instead; for
standing up a fine-tuned checkpoint for the first time, use deploy-checkpoint.

1. If the exact model name isn't known, call list_models (catalog models),
   list_checkpoint_deployments (live checkpoint comparisons), or
   list_finetune_runs (checkpoints that exist but may not be deployed
   anywhere) — don't guess a name.
2. Scaling any existing model up or down (catalog or checkpoint deployment,
   same tool for both): call scale_model(name, 1) to bring it up,
   scale_model(name, 0) to shut it down without deleting anything.
3. Taking down a checkpoint deployment permanently: call
   takedown_checkpoint_model(exp_name) — removes the InferenceService,
   HTTPScaledObject, and the PVCs/Jobs deploy_checkpoint_model created for
   it. This does NOT touch the original fine-tuning checkpoint PVC, so the
   same checkpoint can be redeployed later.
4. Taking down a permanent catalog model (anything under
   platform/base/models/): this repo is GitOps (ArgoCD self-heal + prune) —
   a live delete would just get recreated on the next sync. The only
   durable way is removing its directory from the relevant overlay's
   kustomization.yaml and merging a PR. scale_model(name, 0) is the safe,
   non-destructive way to pause it instead.
