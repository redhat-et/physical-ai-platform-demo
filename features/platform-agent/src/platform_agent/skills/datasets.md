---
name: datasets
description: Use before searching for, inspecting, validating, or pulling a dataset for fine-tuning or general use.
---
DATASETS — in order:

1. list_staged_datasets first, so you don't re-pull an already-staged dataset.
2. For a named robot-policy model, call get_finetune_requirements BEFORE searching — searching for the model's own name returns datasets for any embodiment used with it, not what the recipe needs. Use its query/expected_robot_type with search_compatible_lerobot_datasets, not plain search_datasets. For "a smaller one", use max_size_gb — episode/download count are not a size proxy.
3. get_dataset_info for size, license, and schema — never guess these.
4. To check compatibility with a known model's recipe, call validate_lerobot_dataset(dataset_repo_id=..., model_name=...) rather than passing expected_exterior_cameras/expected_wrist_cameras/expected_action_dim yourself — model_name looks those up directly. NEVER invent an expected_feature_keys value — omit it and read the returned schema yourself if you don't have a real one.
5. EXCEPTION TO RULE 1: never call pull_dataset in the same turn as get_dataset_info — show the user size/license and get explicit go-ahead first.
6. After pulling, use get_dataset_job_status to confirm success before saying the dataset is ready.
