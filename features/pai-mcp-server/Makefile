IMAGE ?= quay.io/redhat-et/pai-mcp-server:latest
CONTAINER_ENGINE ?= docker
PLATFORM ?= linux/amd64

# Local dry-run default: point straight at a live checkout of the skills
# repo instead of the in-cluster git-sync'd volume -- override to test
# against a different checkout, e.g. `make run SKILLS_ROOT=/path/to/skills`.
SKILLS_ROOT ?= $(HOME)/Desktop/physical-ai-platform-agent-skills/skills

.PHONY: build push deploy run

build:
	$(CONTAINER_ENGINE) build --platform $(PLATFORM) -t $(IMAGE) -f Containerfile .

push:
	$(CONTAINER_ENGINE) push $(IMAGE)

deploy: build push
	oc delete pod -n physical-ai -l app=pai-mcp-server

run:
	PAI_MCP_SKILLS_ROOT=$(SKILLS_ROOT) PAI_MCP_OPENSHIFT_MCP_URL=http://localhost:8080/mcp python -m pai_mcp_server
