IMAGE ?= quay.io/redhat-et/physical-ai-agent:latest
CONTAINER_ENGINE ?= docker
PLATFORM ?= linux/amd64
SKILLS_REPO ?= https://github.com/redhat-et/physical-ai-platform-agent-skills.git
SKILLS_REF ?= main

.PHONY: build push deploy run

build:
	$(CONTAINER_ENGINE) build --platform $(PLATFORM) -t $(IMAGE) \
		--build-arg SKILLS_REPO=$(SKILLS_REPO) --build-arg SKILLS_REF=$(SKILLS_REF) \
		--no-cache-filter=skills \
		-f Containerfile .

push:
	$(CONTAINER_ENGINE) push $(IMAGE)

deploy: build push
	oc delete pod -n physical-ai -l app=platform-agent

run:
	uvicorn platform_agent.main:app --host 0.0.0.0 --port 8000 --reload
