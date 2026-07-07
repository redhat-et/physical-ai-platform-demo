IMAGE ?= quay.io/redhat-et/physical-ai-agent:latest

.PHONY: build push run

build:
	podman build -t $(IMAGE) -f Containerfile .

push:
	podman push $(IMAGE)

run:
	uvicorn platform_agent.main:app --host 0.0.0.0 --port 8000 --reload
