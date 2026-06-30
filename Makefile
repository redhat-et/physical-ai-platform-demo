OVERLAYS := dev dev-gpu demo
KUSTOMIZE_BUILD := $(if $(shell command -v kustomize 2>/dev/null),kustomize build,oc kustomize)

.PHONY: validate lint kustomize-build kubeconform

## Run all validation checks
validate: lint kustomize-build kubeconform

## Lint YAML files
lint:
	@echo "=== YAML Lint ==="
	uvx yamllint -c .yamllint.yaml platform/

## Build all kustomize overlays
kustomize-build:
	@for overlay in $(OVERLAYS); do \
		echo "=== kustomize build: $$overlay ==="; \
		$(KUSTOMIZE_BUILD) platform/overlays/$$overlay > /dev/null; \
	done

## Validate manifests against Kubernetes schemas
kubeconform:
	@for overlay in $(OVERLAYS); do \
		echo "=== kubeconform: $$overlay ==="; \
		$(KUSTOMIZE_BUILD) platform/overlays/$$overlay | kubeconform \
			-strict -summary -verbose \
			-kubernetes-version 1.34.0 \
			-schema-location default \
			-schema-location 'https://raw.githubusercontent.com/datreeio/CRDs-catalog/main/{{.Group}}/{{.ResourceKind}}_{{.ResourceAPIVersion}}.json' \
			-ignore-missing-schemas; \
	done
