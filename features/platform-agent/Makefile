IMAGE ?= quay.io/redhat-et/physical-ai-agent:latest
CONTAINER_ENGINE ?= docker
PLATFORM ?= linux/amd64
SKILLS_REPO ?= https://github.com/redhat-et/physical-ai-platform-agent-skills.git
SKILLS_REF ?= main
SKILLS_DIR := src/platform_agent/skills

.PHONY: build push deploy run fetch-skills test

build:
	$(CONTAINER_ENGINE) build --platform $(PLATFORM) -t $(IMAGE) \
		--build-arg SKILLS_REPO=$(SKILLS_REPO) --build-arg SKILLS_REF=$(SKILLS_REF) \
		-f Containerfile .

push:
	$(CONTAINER_ENGINE) push $(IMAGE)

deploy: fetch-skills build push
	oc delete pod -n physical-ai -l app=platform-agent

run:
	uvicorn platform_agent.main:app --host 0.0.0.0 --port 8000 --reload

fetch-skills:
	rm -rf .skills-fetch-tmp
	git clone --depth 1 --branch $(SKILLS_REF) $(SKILLS_REPO) .skills-fetch-tmp
	find $(SKILLS_DIR) -mindepth 1 -maxdepth 1 ! -name '__init__.py' ! -name '__pycache__' -exec rm -rf {} +
	cp -r .skills-fetch-tmp/skills/. $(SKILLS_DIR)/
	rm -rf .skills-fetch-tmp

test: fetch-skills
	pytest
