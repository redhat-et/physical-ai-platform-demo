from langchain_core.tools import tool

# Detailed, step-by-step workflows that are too long to keep inline in the
# system prompt for every single turn -- only loaded into context when the
# agent actually asks for them. Add new workflows here as the agent grows
# into more tool categories, rather than growing the system prompt itself.
_PROCEDURES = {
    "deploy_model": """\
DEPLOY_MODEL — for hardware sizing or adding a new model to the catalog:
1. Call list_cluster_gpus to see real GPU capacity — never guess it.
2. Call estimate_model_footprint with the target Hugging Face repo id to \
get a real recommended tensor_parallel_size — never guess that either.
3. Call generate_model_manifests using the values from steps 1-2.
4. Return the generated YAML to the user verbatim in fenced code blocks — \
do not paraphrase, shorten, or summarize it.
5. Tell the user this is a draft only: this platform uses GitOps (ArgoCD \
self-heal + prune), so nothing is actually deployed until a human saves \
the files, wires them into an overlay, and merges a PR. You cannot deploy \
a model yourself.\
""",
}


@tool
def get_procedure(topic: str) -> str:
    """Look up detailed step-by-step instructions for a specific workflow
    that isn't fully covered in your base instructions. Call this before
    attempting an unfamiliar multi-step task rather than guessing the
    steps yourself.

    Args:
        topic: The workflow to look up. Currently supported: 'deploy_model'
            (hardware sizing and adding a new model to the catalog).
    """
    procedure = _PROCEDURES.get(topic)
    if procedure is None:
        return (
            f"No procedure found for '{topic}'. Available topics: "
            f"{sorted(_PROCEDURES.keys())}."
        )
    return procedure
