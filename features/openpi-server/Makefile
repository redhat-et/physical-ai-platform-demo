IMAGE ?= quay.io/redhat-et/openpi-server:latest
CONTAINER_ENGINE ?= podman
PLATFORM ?= linux/amd64

.PHONY: build push deploy

build:
	$(CONTAINER_ENGINE) build --platform $(PLATFORM) -t $(IMAGE) -f Containerfile .

push:
	$(CONTAINER_ENGINE) push $(IMAGE)

deploy: build push
