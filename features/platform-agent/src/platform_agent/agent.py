import httpx
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from openai import DefaultHttpxClient

from platform_agent.config import settings
from platform_agent.tools.inference import call_model
from platform_agent.tools.hardware import list_cluster_gpus, estimate_model_footprint
from platform_agent.tools.manifests import generate_model_manifests
from platform_agent.tools.datasets import (
    get_dataset_info,
    get_dataset_rows,
    get_dataset_file,
    validate_dataset_schema,
    validate_lerobot_dataset,
    pull_dataset,
    get_dataset_job_status,
    convert_dataset_to_v3,
    get_dataset_conversion_status,
    list_staged_datasets,
)
from platform_agent.tools.finetune import (
    submit_finetune_run,
    get_finetune_run_status,
    list_finetune_runs,
)
from platform_agent.tools.checkpoint_deploy import (
    deploy_checkpoint_model,
    get_checkpoint_deployment_status,
    takedown_checkpoint_model,
    list_checkpoint_deployments,
)
from platform_agent.tools.skills import get_skill, list_skills

# Deduped, order-preserving -- infra_namespace and datasets_namespace are
# both "physical-ai" today but aren't guaranteed to stay that way, and this
# drives RULE 5's namespace list, so it must reflect the real settings
# rather than being hardcoded separately in the prompt text.
_NAMESPACES = ", ".join(dict.fromkeys([
    settings.models_namespace,
    settings.infra_namespace,
    settings.maas_namespace,
    settings.datasets_namespace,
]))

SYSTEM_PROMPT = (
    settings.system_prompt
    .replace("{model}", settings.llm_model)
    .replace("{namespaces}", _NAMESPACES)
)

TOOLS = [
    call_model,
    list_cluster_gpus,
    estimate_model_footprint,
    generate_model_manifests,
    get_dataset_info,
    get_dataset_rows,
    get_dataset_file,
    validate_dataset_schema,
    validate_lerobot_dataset,
    pull_dataset,
    get_dataset_job_status,
    convert_dataset_to_v3,
    get_dataset_conversion_status,
    list_staged_datasets,
    submit_finetune_run,
    get_finetune_run_status,
    list_finetune_runs,
    deploy_checkpoint_model,
    get_checkpoint_deployment_status,
    takedown_checkpoint_model,
    list_checkpoint_deployments,
    get_skill,
    list_skills,
]


def build_agent(use_tools: bool = True, extra_tools: list = ()):
    llm = ChatOpenAI(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        temperature=0,
        http_client=DefaultHttpxClient(verify=False),
        http_async_client=httpx.AsyncClient(verify=False),
    )

    if use_tools:
        try:
            agent = create_react_agent(llm, [*TOOLS, *extra_tools], prompt=SYSTEM_PROMPT)
            return ("agent", agent)
        except Exception:
            pass

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])
    chain = prompt | llm
    return ("chain", chain)
