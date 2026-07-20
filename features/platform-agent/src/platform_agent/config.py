from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    models_namespace: str = "physical-ai-models"
    infra_namespace: str = "physical-ai"
    maas_namespace: str = "models-as-a-service"
    datasets_namespace: str = "physical-ai"

    llm_base_url: str = "http://maas-proxy.physical-ai.svc.cluster.local:8080/v1"
    llm_model: str = "qwen3-omni"
    llm_api_key: str = "unused"
    maas_proxy_url: str = "https://maas-proxy.physical-ai-models.svc.cluster.local"

    # Set via the AGENT_SYSTEM_PROMPT env var, sourced from the
    # platform-agent-config ConfigMap -- that ConfigMap is the only copy of
    # this text, so it can be edited live without rebuilding the agent image.
    # May contain the literal placeholders {model} and {ns}, substituted in
    # agent.py.
    system_prompt: str

    model_config = {"env_prefix": "AGENT_"}


settings = Settings()
