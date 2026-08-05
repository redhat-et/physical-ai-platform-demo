from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8081

    # Kept current on this path by a git-sync sidecar sharing a volume with
    # this container (see Phase 3 manifests) -- never baked into this
    # server's own image, so a skill update takes effect on the next sync
    # interval with no rebuild or redeploy of pai-mcp-server itself. For
    # local dry-runs, point this at a local skills-repo checkout instead.
    skills_root: str = "/skills"

    # openshift-mcp-server runs as this server's own sidecar (see Phase 3
    # manifests) -- reachable on localhost, never dialed directly by agents.
    openshift_mcp_url: str = "http://localhost:8080/mcp"
    k8s_tool_prefix: str = "k8s_"

    script_timeout_seconds: int = 300

    model_config = {"env_prefix": "PAI_MCP_"}


settings = Settings()
