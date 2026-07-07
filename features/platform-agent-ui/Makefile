IMAGE ?= quay.io/redhat-et/platform-agent-ui:latest

.PHONY: build push

build:
	podman build -t $(IMAGE) -f Containerfile .

push:
	podman push $(IMAGE)
