IMAGE ?= quay.io/redhat-et/physical-ai-agent-ui:latest
CONTAINER_ENGINE ?= docker
PLATFORM ?= linux/amd64

.PHONY: build push deploy

build:
	$(CONTAINER_ENGINE) build --platform $(PLATFORM) -t $(IMAGE) -f Containerfile .

push:
	$(CONTAINER_ENGINE) push $(IMAGE)

deploy: build push
	oc delete pod -n redhat-ods-applications -l app=platform-agent-ui
