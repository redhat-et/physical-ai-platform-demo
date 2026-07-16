from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    models_namespace: str = "physical-ai-models"
    infra_namespace: str = "physical-ai"
    maas_namespace: str = "models-as-a-service"
    # Same namespace as the DSPA (infra_namespace): fine-tuning pipeline
    # pods run there, and a PVC can only be mounted by a pod in its own
    # namespace, so dataset/checkpoint PVCs have to live here too.
    datasets_namespace: str = "physical-ai"

    llm_base_url: str = "http://maas-proxy.physical-ai.svc.cluster.local:8080/v1"
    llm_model: str = "qwen3-omni"
    llm_api_key: str = "unused"
    maas_proxy_url: str = "https://maas-proxy.physical-ai-models.svc.cluster.local"

    model_config = {"env_prefix": "AGENT_"}


settings = Settings()
