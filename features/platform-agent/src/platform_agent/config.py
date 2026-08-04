from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    models_namespace: str = "physical-ai-models"
    infra_namespace: str = "physical-ai"
    maas_namespace: str = "models-as-a-service"
    datasets_namespace: str = "physical-ai"

<<<<<<< Updated upstream
=======
    skills_root: str = str(Path(__file__).resolve().parent / "skills")

>>>>>>> Stashed changes
    llm_base_url: str = "http://maas-proxy.physical-ai.svc.cluster.local:8080/v1"
    llm_model: str = "qwen3-omni"
    llm_api_key: str = "unused"
    maas_proxy_url: str = "https://maas-proxy.physical-ai-models.svc.cluster.local"

    model_catalog_raw_base: str = "https://raw.githubusercontent.com/redhat-et/physical-ai-platform-demo/main"

    system_prompt: str

    model_config = {"env_prefix": "AGENT_"}


settings = Settings()
