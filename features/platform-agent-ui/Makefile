IMAGE ?= quay.io/redhat-et/physical-ai-agent-ui:latest

.PHONY: build push

build:
	podman build -t $(IMAGE) -f Containerfile .

push:
	podman push $(IMAGE)
