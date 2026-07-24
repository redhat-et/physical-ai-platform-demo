---
name: deploy-checkpoint
description: Use to stand up a finished fine-tuning checkpoint as a live model endpoint for testing or comparison against the base model.
---
DEPLOY CHECKPOINT — for standing up a fine-tuned checkpoint as a real,
callable model endpoint, in order:

1. Confirm the fine-tuning run actually succeeded first — call
   get_finetune_run_status(exp_name) (or list_finetune_runs if the exact
   exp_name isn't known). Don't deploy a checkpoint from a run that's still
   in progress or failed. Only 'pi05' is supported (same restriction as the
   fine-tuning skill).
2. Call deploy_checkpoint_model(exp_name). This is a LIVE action, not a
   GitOps draft like the deploy-model skill — it copies the checkpoint into
   the models namespace, converting it along the way into a format
   openpi-runtime can serve, and reuses the base pi05_droid checkpoint's own
   normalization stats (not stats recomputed from this run's dataset --
   mention that if asked how trustworthy the deployed checkpoint's
   predictions are). It's ephemeral by design: never committed to git, so it
   won't show up in a PR and ArgoCD will never touch it.
3. Call get_checkpoint_deployment_status(exp_name) repeatedly to advance and
   check progress — unlike get_finetune_run_status, nothing else drives this
   forward on its own; each call both reports status and, once the current
   stage is ready, kicks off the next one. Keep calling it until it reports
   the model deployed. Copying a full checkpoint takes a couple of minutes,
   so "still copying" on an early poll is expected, not a failure.
4. The deployed model is scale-to-zero, same as catalog models — call
   scale_model(isvc_name, 1) to actually warm it up before testing or
   calling it, using the isvc_name reported by get_checkpoint_deployment_status.
   First startup can take several minutes (image pull plus the server's own
   warmup inference) — use get_model_status(isvc_name) and, if it's taking a
   while, get_pod_logs to check real progress rather than assuming failure.
5. "Deployed" only means the InferenceService/pod came up healthy -- it does
   not mean a real inference request has been verified. If asked to confirm
   the checkpoint actually serves correctly, that needs an actual request
   against the predictor (or a WebSocket handshake as a lighter check), not
   just a healthy pod.
6. This is NOT yet wired into the robotics playground's model list — that's
   a manual follow-up, not something this skill does automatically.
7. When done comparing, use the manage-models skill's
   takedown_checkpoint_model to tear it down and free the GPU/storage it
   was using.
