---
name: fine-tuning
description: Use before discussing, submitting, or checking on a fine-tuning run.
---
FINE-TUNING — in order:

1. Confirm the dataset is staged (pull_dataset) and, for robot-policy models, validated (validate_lerobot_dataset) before ever calling submit_finetune_run — see the datasets skill for that workflow. If the dataset repo bundles multiple independent LeRobot datasets as subfolders (see the datasets skill), pass the subfolder name as submit_finetune_run's dataset_subset.
2. Discuss the recipe with the user first — model, dataset, that this runs real GPU-hours on the shared cluster for potentially hours.
3. EXCEPTION TO RULE 1: never call submit_finetune_run in the same turn as the initial fine-tuning request, for any reason, including to "check" whether the dataset/config is valid — that speculative call IS the forbidden action, whether or not it succeeds. Use get_dataset_job_status / list_staged_datasets instead to check preconditions. Wait until the user explicitly says to proceed — same carve-out as pull_dataset, higher stakes (GPU-hours, not just storage).
4. The pipeline advances through its own stages on its own — get_finetune_run_status is a read-only progress check, not something that needs repeated calls to make a stage happen.
5. If the exact exp_name isn't known, or the question is general ("what's running", "any fine-tunes in progress"), call list_finetune_runs — don't guess a name or say there's no way to check.
6. Relay final eval numbers and the checkpoint PVC name only once get_finetune_run_status reports all stages complete.
7. If get_finetune_run_status reports a FAILED stage, it now includes that stage's recent log lines -- read them. If they mention a LeRobot dataset-format/version error (e.g. BackwardCompatibilityError, "v2.1", "v3.0"): call convert_dataset_to_v3(dataset_pvc_name), wait for get_dataset_conversion_status to report succeeded, then call submit_finetune_run again with the same dataset_pvc_name and exp_name -- resubmitting under the same exp_name is only allowed once the prior run's state is FAILED. For any other failure reason, don't guess a fix -- report the log contents to the user.
