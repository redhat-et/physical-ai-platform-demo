IMAGE ?= quay.io/redhat-et/physical-ai-platform-robotics-playground
TAG   ?= latest

.PHONY: build push

build:
	podman build -t $(IMAGE):$(TAG) -f Containerfile .

push:
	podman push $(IMAGE):$(TAG)
