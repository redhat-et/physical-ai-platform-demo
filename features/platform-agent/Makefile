IMAGE ?= quay.io/redhat-et/physical-ai-agent:latest
CONTAINER_ENGINE ?= docker
PLATFORM ?= linux/amd64

.PHONY: build push deploy run

build:
	$(CONTAINER_ENGINE) build --platform $(PLATFORM) -t $(IMAGE) -f Containerfile .

push:
	$(CONTAINER_ENGINE) push $(IMAGE)

deploy: build push
	oc delete pod -n physical-ai -l app=platform-agent

run:
	uvicorn platform_agent.main:app --host 0.0.0.0 --port 8000 --reload
