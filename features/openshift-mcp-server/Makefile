IMAGE ?= quay.io/redhat-et/openshift-mcp-server:latest
CONTAINER_ENGINE ?= docker
PLATFORM ?= linux/amd64
MCP_SERVER_REPO ?= https://github.com/openshift/openshift-mcp-server.git
MCP_SERVER_REF ?= main

.PHONY: build push deploy

build:
	$(CONTAINER_ENGINE) build --platform $(PLATFORM) -t $(IMAGE) \
		--build-arg MCP_SERVER_REPO=$(MCP_SERVER_REPO) --build-arg MCP_SERVER_REF=$(MCP_SERVER_REF) \
		-f Containerfile .

push:
	$(CONTAINER_ENGINE) push $(IMAGE)

deploy: build push
	oc delete pod -n physical-ai -l app=platform-agent
