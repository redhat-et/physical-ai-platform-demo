# Physical AI Platform Demo — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the repo structure, preflight script, and kustomize overlays for deploying the Physical AI Platform on any OpenShift cluster with RHOAI 3.4+.

**Architecture:** Kustomize base + overlays pattern. A shared base defines MaaS infrastructure and model registrations. Three overlays (`dev`, `dev-gpu`, `demo`) adjust resource allocation and model backends. A preflight script validates prerequisites before deployment. Documentation is split into a top-level README (quickstart) and `platform/prerequisites.md` (detailed install guidance).

**Tech Stack:** Kustomize, Bash (`oc` CLI)

**Reference:** Design spec at `docs/superpowers/specs/2026-06-22-physical-ai-platform-design.md`, MaaS research notes at `notes/research/rhoai-maas-production-stack.md`.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `README.md` | Project overview, quickstart, environment profiles |
| `platform/prerequisites.md` | Detailed prerequisites with versions and install commands |
| `platform/preflight.sh` | Read-only prerequisite validation script |
| `platform/base/kustomization.yaml` | Root kustomize base referencing sub-bases |
| `platform/base/namespace.yaml` | Platform namespaces (`physical-ai`, `physical-ai-models`) |
| `platform/base/dsc-patch.yaml` | Strategic merge patch for DataScienceCluster |
| `platform/base/maas/kustomization.yaml` | MaaS sub-base aggregator |
| `platform/base/maas/gateway.yaml` | `maas-default-gateway` Gateway resource |
| `platform/base/maas/tenant.yaml` | Default Tenant CR in `models-as-a-service` namespace |
| `platform/base/maas/postgresql.yaml` | Dev PostgreSQL StatefulSet + `maas-db-config` Secret |
| `platform/base/models/kustomization.yaml` | Models sub-base aggregator |
| `platform/base/models/mocklm/kustomization.yaml` | Mock LM model sub-base |
| `platform/base/models/mocklm/deployment.yaml` | mocklm Deployment + Service |
| `platform/base/models/mocklm/external-model.yaml` | ExternalModel CR + credential Secret |
| `platform/base/models/mocklm/model-ref.yaml` | MaaSModelRef CR |
| `platform/base/models/mocklm/subscription.yaml` | MaaSSubscription CR |
| `platform/base/models/mocklm/auth-policy.yaml` | MaaSAuthPolicy CR |
| `platform/overlays/dev/kustomization.yaml` | Dev profile overlay (uses base as-is) |
| `platform/overlays/dev-gpu/kustomization.yaml` | GPU profile overlay (placeholder for real models) |
| `platform/overlays/demo/kustomization.yaml` | Demo profile overlay (placeholder for full model set) |

---

### Task 1: README and prerequisites documentation

**Files:**

- Create: `README.md`
- Create: `platform/prerequisites.md`

- [ ] **Step 1: Create README.md**

```markdown
# Physical AI Platform Demo

A proof-of-concept platform extending Red Hat OpenShift AI (RHOAI) to the
domain of physical AI — robotics, autonomous vehicles, digital twins.

Deploys the production RHOAI Models-as-a-Service (MaaS) stack with
Kuadrant/Red Hat Connectivity Link and registers physical AI models through
the governed gateway.

## Quickstart

```bash
# 1. Validate prerequisites
./platform/preflight.sh

# 2. Deploy (pick one profile)
oc apply -k platform/overlays/dev        # no GPU, mock models
oc apply -k platform/overlays/dev-gpu    # with GPU, real model serving
oc apply -k platform/overlays/demo       # full demo environment
```

See [platform/prerequisites.md](platform/prerequisites.md) for what must be
installed before deploying.

## Environment Profiles

| Profile   | GPU | Models                         | Use Case                    |
|-----------|-----|--------------------------------|-----------------------------|
| dev       | No  | Mock servers / external proxy  | Development, CI, testing    |
| dev-gpu   | Yes | Real model serving (vLLM)      | Development with inference  |
| demo      | Yes | Full model set                 | Demonstrations, workshops   |

The architecture is identical across profiles. The difference is what model
backends are registered (mock vs. real) and resource allocation. Any cluster
meeting the prerequisites can run any profile.

## Repository Layout

```
physical-ai-platform-demo/
├── platform/                  # deployable manifests
│   ├── prerequisites.md       # what must be installed first
│   ├── preflight.sh           # validates prerequisites
│   ├── base/                  # shared kustomize base
│   │   ├── maas/              # MaaS gateway, PostgreSQL
│   │   └── models/            # model registrations
│   │       └── mocklm/        # mock LM (OpenAI-compatible)
│   └── overlays/              # per-environment overlays
│       ├── dev/               # minimal, no GPU
│       ├── dev-gpu/           # GPU-enabled
│       └── demo/              # full demo
├── docs/                      # design specs and plans
└── notes/                     # gitignored: local snapshots, research
```

## Removing the Platform

```bash
oc delete -k platform/overlays/dev  # same overlay used to deploy
```

```

- [ ] **Step 2: Create directory and write platform/prerequisites.md**

```bash
mkdir -p platform
```

Write `platform/prerequisites.md`:

```markdown
# Prerequisites

The Physical AI Platform requires the following operators and cluster
capabilities before deployment. The platform does not install these —
they must be set up by a cluster admin.

## Required Operators

| Operator | Min Version | OperatorHub Package | Purpose |
|----------|-------------|---------------------|---------|
| OpenShift Container Platform | 4.19.9 | — | Gateway API CRDs (native in 4.19+) |
| Red Hat OpenShift AI | 3.4 | `rhods-operator` | KServe, MaaS controller, Dashboard |
| Red Hat Connectivity Link | 1.3 | `kuadrant-operator` | Kuadrant, Authorino, Limitador |
| cert-manager | 1.x | `openshift-cert-manager-operator` | TLS certificates |

## Required Cluster Configuration

- **Default StorageClass** — any CSI provisioner with a default StorageClass.
  Verify with: `oc get sc`
- **User Workload Monitoring** — must be enabled for MaaS metrics.

## Installation Guidance

### RHOAI 3.4

Install from OperatorHub (Installed Operators → Red Hat OpenShift AI) or via
CLI:

```bash
oc create namespace redhat-ods-operator 2>/dev/null || true
cat <<'EOF' | oc apply -f -
apiVersion: operators.coreos.com/v1alpha1
kind: Subscription
metadata:
  name: rhods-operator
  namespace: redhat-ods-operator
spec:
  channel: stable-3.4
  name: rhods-operator
  source: redhat-operators
  sourceNamespace: openshift-marketplace
  installPlanApproval: Automatic
EOF
```

Wait for the operator pod to be ready, then create the DSCInitialization (if
one doesn't already exist):

```bash
cat <<'EOF' | oc apply -f -
apiVersion: dscinitialization.opendatahub.io/v1
kind: DSCInitialization
metadata:
  name: default-dsci
spec:
  applicationsNamespace: redhat-ods-applications
  monitoring:
    managementState: Managed
    namespace: redhat-ods-monitoring
EOF
```

The DataScienceCluster CR is created by the platform's kustomize manifests —
do **not** create one manually.

### Red Hat Connectivity Link 1.3

```bash
cat <<'EOF' | oc apply -f -
apiVersion: operators.coreos.com/v1alpha1
kind: Subscription
metadata:
  name: kuadrant-operator
  namespace: openshift-operators
spec:
  channel: stable
  name: kuadrant-operator
  source: redhat-operators
  sourceNamespace: openshift-marketplace
  installPlanApproval: Automatic
EOF
```

After the operator is ready, create the Kuadrant CR:

```bash
oc create namespace kuadrant-system 2>/dev/null || true
cat <<'EOF' | oc apply -f -
apiVersion: kuadrant.io/v1beta1
kind: Kuadrant
metadata:
  name: kuadrant
  namespace: kuadrant-system
EOF
```

### User Workload Monitoring

```bash
cat <<'EOF' | oc apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: cluster-monitoring-config
  namespace: openshift-monitoring
data:
  config.yaml: |
    enableUserWorkload: true
EOF
```

## Validation

Run the preflight script to check all prerequisites:

```bash
./platform/preflight.sh
```

```

- [ ] **Step 3: Commit**

```bash
git add README.md platform/prerequisites.md
git commit -m "docs: add README and prerequisites documentation"
```

---

### Task 2: Preflight script

**Files:**

- Create: `platform/preflight.sh`

- [ ] **Step 1: Write platform/preflight.sh**

```bash
#!/usr/bin/env bash
set -euo pipefail

PASS=0
FAIL=0
WARN=0

check_pass() { echo "  ✓ $*"; PASS=$((PASS + 1)); }
check_fail() { echo "  ✗ $*"; FAIL=$((FAIL + 1)); }
check_warn() { echo "  ! $*"; WARN=$((WARN + 1)); }

echo "Physical AI Platform — Preflight Check"
echo "========================================"
echo ""

# --- Cluster access ---
echo "Cluster access:"
if ! command -v oc &>/dev/null; then
    check_fail "oc CLI not found in PATH"
    echo ""
    echo "Result: $FAIL FAILED — install the oc CLI first"
    exit 1
fi

if ! oc whoami &>/dev/null; then
    check_fail "Not logged in — run 'oc login' first"
    echo ""
    echo "Result: $FAIL FAILED — cannot continue without cluster access"
    exit 1
fi
check_pass "Logged in as $(oc whoami) to $(oc whoami --show-server)"

if oc auth can-i '*' '*' --all-namespaces &>/dev/null; then
    check_pass "cluster-admin privileges"
else
    check_fail "cluster-admin privileges required"
fi
echo ""

# --- OCP version ---
echo "OpenShift version:"
OCP_VERSION="$(oc get clusterversion version -o jsonpath='{.status.desired.version}' 2>/dev/null || echo 'unknown')"
OCP_MAJOR_MINOR="$(echo "$OCP_VERSION" | awk -F. '{printf "%s.%s", $1, $2}')"
if [[ "$OCP_VERSION" == "unknown" ]]; then
    check_fail "Could not determine OpenShift version"
elif awk "BEGIN{exit !($OCP_MAJOR_MINOR >= 4.19)}"; then
    check_pass "OpenShift $OCP_VERSION (>= 4.19 required)"
else
    check_fail "OpenShift $OCP_VERSION — version 4.19+ required for Gateway API"
fi
echo ""

# --- Required operators ---
echo "Required operators:"

find_csv_version() {
    local prefix="$1"
    oc get csv -A -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.version}{"\n"}{end}' 2>/dev/null \
        | awk -F'\t' -v prefix="$prefix" '$1 ~ "^"prefix { print $2; exit }'
}

check_operator() {
    local label="$1"
    local csv_prefix="$2"
    local min_version="$3"
    local version
    version="$(find_csv_version "$csv_prefix")"
    if [[ -n "$version" ]]; then
        check_pass "$label $version (>= $min_version required)"
    else
        check_fail "$label not installed (>= $min_version required) — see platform/prerequisites.md"
    fi
}

check_operator "Red Hat OpenShift AI" "rhods-operator" "3.4"
check_operator "Red Hat Connectivity Link" "kuadrant-operator" "1.3"
check_operator "cert-manager Operator" "cert-manager-operator" "1.0"
echo ""

# --- Kuadrant CR ---
echo "Kuadrant:"
if oc get kuadrant -n kuadrant-system &>/dev/null 2>&1; then
    check_pass "Kuadrant CR exists in kuadrant-system"
else
    check_fail "Kuadrant CR not found in kuadrant-system — see platform/prerequisites.md"
fi
echo ""

# --- Storage ---
echo "Storage:"
DEFAULT_SC="$(oc get sc -o jsonpath='{.items[?(@.metadata.annotations.storageclass\.kubernetes\.io/is-default-class=="true")].metadata.name}' 2>/dev/null)"
if [[ -n "$DEFAULT_SC" ]]; then
    check_pass "Default StorageClass: $DEFAULT_SC"
else
    check_fail "No default StorageClass — a CSI provisioner with a default StorageClass is required"
fi
echo ""

# --- User Workload Monitoring ---
echo "Monitoring:"
UWM_CONFIG="$(oc get configmap cluster-monitoring-config -n openshift-monitoring -o jsonpath='{.data.config\.yaml}' 2>/dev/null || true)"
if echo "$UWM_CONFIG" | grep -q 'enableUserWorkload: true'; then
    check_pass "User Workload Monitoring enabled"
else
    check_warn "User Workload Monitoring may not be enabled — MaaS metrics will be unavailable"
fi
echo ""

# --- Summary ---
echo "========================================"
echo "Result: $PASS passed, $FAIL failed, $WARN warnings"
if [[ "$FAIL" -gt 0 ]]; then
    echo ""
    echo "Fix the failures above before deploying."
    echo "See platform/prerequisites.md for installation guidance."
    exit 1
fi
```

- [ ] **Step 2: Make executable and verify syntax**

```bash
chmod +x platform/preflight.sh
bash -n platform/preflight.sh
```

Expected: no output (syntax OK).

- [ ] **Step 3: Commit**

```bash
git add platform/preflight.sh
git commit -m "feat: add preflight script to validate platform prerequisites"
```

---

### Task 3: Kustomize base — namespaces and DSC patch

**Files:**

- Create: `platform/base/kustomization.yaml`
- Create: `platform/base/namespace.yaml`
- Create: `platform/base/dsc-patch.yaml`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p platform/base
```

- [ ] **Step 2: Create platform/base/namespace.yaml**

Three namespaces: `physical-ai` for platform infrastructure (PostgreSQL), `physical-ai-models` for model deployments and registrations, `models-as-a-service` for MaaS tenant configuration (subscriptions, auth policies).

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: physical-ai
---
apiVersion: v1
kind: Namespace
metadata:
  name: physical-ai-models
---
apiVersion: v1
kind: Namespace
metadata:
  name: models-as-a-service
```

- [ ] **Step 3: Create platform/base/dsc-patch.yaml**

Strategic merge patch for the existing DataScienceCluster. Enables MaaS and sets `rawDeploymentServiceConfig: Headed` (required by MaaS). The DSC named `default-dsc` is the convention used by RHOAI — if the cluster has a differently-named DSC, the overlay's kustomization target must be adjusted.

```yaml
apiVersion: datasciencecluster.opendatahub.io/v2
kind: DataScienceCluster
metadata:
  name: default-dsc
spec:
  components:
    kserve:
      managementState: Managed
      rawDeploymentServiceConfig: Headed
      modelsAsService:
        managementState: Managed
    dashboard:
      managementState: Managed
```

- [ ] **Step 4: Create platform/base/kustomization.yaml**

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - namespace.yaml
  - maas/
  - models/

patches:
  - path: dsc-patch.yaml
    target:
      group: datasciencecluster.opendatahub.io
      version: v2
      kind: DataScienceCluster
      name: default-dsc
```

- [ ] **Step 5: Verify kustomize renders the namespace resources**

At this point `maas/` and `models/` don't exist yet, so we verify the namespace file in isolation:

```bash
oc kustomize platform/base/ 2>&1 || echo "(expected — maas/ and models/ not yet created)"
```

Expected: error about missing `maas/` directory. This is correct — we'll add it in Task 4.

- [ ] **Step 6: Commit**

```bash
git add platform/base/kustomization.yaml platform/base/namespace.yaml platform/base/dsc-patch.yaml
git commit -m "feat: add kustomize base with namespaces and DSC patch"
```

---

### Task 4: Kustomize base — MaaS infrastructure

**Files:**

- Create: `platform/base/maas/kustomization.yaml`
- Create: `platform/base/maas/gateway.yaml`
- Create: `platform/base/maas/tenant.yaml`
- Create: `platform/base/maas/postgresql.yaml`

- [ ] **Step 1: Create directory**

```bash
mkdir -p platform/base/maas
```

- [ ] **Step 2: Create platform/base/maas/gateway.yaml**

The MaaS gateway in `openshift-ingress`. Uses `openshift-default` gatewayClassName which is available on OCP 4.19+. The TLS secret `maas-gateway-tls` must exist in `openshift-ingress` — in development, cert-manager can provision it; for now the reference is a placeholder that the MaaS controller may auto-provision.

```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: maas-default-gateway
  namespace: openshift-ingress
spec:
  gatewayClassName: openshift-default
  listeners:
    - name: maas
      port: 443
      protocol: HTTPS
      tls:
        mode: Terminate
        certificateRefs:
          - name: maas-gateway-tls
      allowedRoutes:
        namespaces:
          from: All
```

- [ ] **Step 3: Create platform/base/maas/tenant.yaml**

The default Tenant CR bootstraps the `models-as-a-service` namespace for MaaSSubscription and MaaSAuthPolicy resources. This namespace is a MaaS convention — subscriptions and auth policies live here, separate from model deployments.

```yaml
apiVersion: maas.opendatahub.io/v1alpha1
kind: Tenant
metadata:
  name: default-tenant
  namespace: models-as-a-service
```

- [ ] **Step 4: Create platform/base/maas/postgresql.yaml**

Minimal dev-grade PostgreSQL for MaaS API key storage. The `maas-db-config` Secret goes in `redhat-ods-applications` where the MaaS controller expects it.

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: maas-db-credentials
  namespace: physical-ai
type: Opaque
stringData:
  POSTGRESQL_USER: maas
  POSTGRESQL_PASSWORD: maas-dev-password
  POSTGRESQL_DATABASE: maas
---
apiVersion: v1
kind: Service
metadata:
  name: maas-db
  namespace: physical-ai
spec:
  selector:
    app: maas-db
  ports:
    - port: 5432
      targetPort: 5432
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: maas-db
  namespace: physical-ai
  labels:
    app: maas-db
spec:
  serviceName: maas-db
  replicas: 1
  selector:
    matchLabels:
      app: maas-db
  template:
    metadata:
      labels:
        app: maas-db
    spec:
      containers:
        - name: postgresql
          image: registry.redhat.io/rhel9/postgresql-15:latest
          ports:
            - containerPort: 5432
          envFrom:
            - secretRef:
                name: maas-db-credentials
          resources:
            requests:
              cpu: 50m
              memory: 128Mi
            limits:
              cpu: 200m
              memory: 256Mi
          volumeMounts:
            - name: data
              mountPath: /var/lib/pgsql/data
          readinessProbe:
            exec:
              command: ["pg_isready", "-U", "maas"]
            initialDelaySeconds: 10
            periodSeconds: 5
  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes: [ReadWriteOnce]
        resources:
          requests:
            storage: 1Gi
---
apiVersion: v1
kind: Secret
metadata:
  name: maas-db-config
  namespace: redhat-ods-applications
type: Opaque
stringData:
  DB_CONNECTION_URL: "postgresql://maas:maas-dev-password@maas-db.physical-ai.svc:5432/maas?sslmode=disable"
```

- [ ] **Step 5: Create platform/base/maas/kustomization.yaml**

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - gateway.yaml
  - tenant.yaml
  - postgresql.yaml
```

- [ ] **Step 6: Verify kustomize renders the MaaS resources**

```bash
oc kustomize platform/base/ 2>&1 || echo "(expected — models/ not yet created)"
```

Expected: error about missing `models/` directory, but the maas resources should be parseable.

- [ ] **Step 7: Commit**

```bash
git add platform/base/maas/
git commit -m "feat: add MaaS infrastructure — gateway, tenant, and dev PostgreSQL"
```

---

### Task 5: Kustomize base — mock LM model registration

**Files:**

- Create: `platform/base/models/kustomization.yaml`
- Create: `platform/base/models/mocklm/kustomization.yaml`
- Create: `platform/base/models/mocklm/deployment.yaml`
- Create: `platform/base/models/mocklm/external-model.yaml`
- Create: `platform/base/models/mocklm/model-ref.yaml`
- Create: `platform/base/models/mocklm/subscription.yaml`
- Create: `platform/base/models/mocklm/auth-policy.yaml`

- [ ] **Step 1: Create directories**

```bash
mkdir -p platform/base/models/mocklm
```

- [ ] **Step 2: Create platform/base/models/mocklm/deployment.yaml**

Deploy mocklm as a plain Deployment + Service. Not using KServe — this is a mock, not a real model. The `MOCKLM_MODEL_NAME` env var controls what name appears in `/v1/models` responses.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mocklm
  namespace: physical-ai-models
  labels:
    app: mocklm
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mocklm
  template:
    metadata:
      labels:
        app: mocklm
    spec:
      containers:
        - name: mocklm
          image: quay.io/fzdarsky/mocklm:latest
          ports:
            - containerPort: 8080
              protocol: TCP
          env:
            - name: MOCKLM_MODE
              value: echo
            - name: MOCKLM_MODEL_NAME
              value: physical-ai-lm
            - name: MOCKLM_CATCH_ALL
              value: "true"
          resources:
            requests:
              cpu: 50m
              memory: 64Mi
            limits:
              cpu: 200m
              memory: 128Mi
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 3
            periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: mocklm
  namespace: physical-ai-models
spec:
  selector:
    app: mocklm
  ports:
    - port: 8080
      targetPort: 8080
      protocol: TCP
```

- [ ] **Step 3: Create platform/base/models/mocklm/external-model.yaml**

Register the in-cluster mock LM as an ExternalModel with MaaS. The `provider: openai` tells the MaaS gateway this endpoint speaks the OpenAI protocol (which mocklm does). The credential Secret uses the `inference.networking.k8s.io/bbr-managed` label required by the ExternalModel reconciler.

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: mocklm-credentials
  namespace: physical-ai-models
  labels:
    inference.networking.k8s.io/bbr-managed: "true"
type: Opaque
stringData:
  api-key: mock
---
apiVersion: maas.opendatahub.io/v1alpha1
kind: ExternalModel
metadata:
  name: physical-ai-lm
  namespace: physical-ai-models
spec:
  provider: openai
  endpoint: mocklm.physical-ai-models.svc.cluster.local:8080
  targetModel: physical-ai-lm
  credentialRef:
    name: mocklm-credentials
```

- [ ] **Step 4: Create platform/base/models/mocklm/model-ref.yaml**

```yaml
apiVersion: maas.opendatahub.io/v1alpha1
kind: MaaSModelRef
metadata:
  name: physical-ai-lm
  namespace: physical-ai-models
spec:
  modelRef:
    kind: ExternalModel
    name: physical-ai-lm
```

- [ ] **Step 5: Create platform/base/models/mocklm/subscription.yaml**

A default subscription with generous limits for development.

```yaml
apiVersion: maas.opendatahub.io/v1alpha1
kind: MaaSSubscription
metadata:
  name: physical-ai-dev
  namespace: models-as-a-service
spec:
  models:
    - name: physical-ai-lm
      namespace: physical-ai-models
  tokenRateLimit:
    requestsPerMinute: 100
    tokensPerMinute: 100000
```

- [ ] **Step 6: Create platform/base/models/mocklm/auth-policy.yaml**

Allow all authenticated users access to the model for development.

```yaml
apiVersion: maas.opendatahub.io/v1alpha1
kind: MaaSAuthPolicy
metadata:
  name: physical-ai-dev
  namespace: models-as-a-service
spec:
  models:
    - name: physical-ai-lm
      namespace: physical-ai-models
  groups:
    - system:authenticated
```

- [ ] **Step 7: Create platform/base/models/mocklm/kustomization.yaml**

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - deployment.yaml
  - external-model.yaml
  - model-ref.yaml
  - subscription.yaml
  - auth-policy.yaml
```

- [ ] **Step 8: Create platform/base/models/kustomization.yaml**

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - mocklm/
```

- [ ] **Step 9: Verify full base renders**

```bash
oc kustomize platform/base/
```

Expected: all resources render — 3 Namespaces, 1 Gateway, 1 Tenant, 1 StatefulSet, 2 Secrets (db-credentials, maas-db-config), 1 Service (maas-db), 1 Deployment (mocklm), 1 Service (mocklm), 1 Secret (mocklm-credentials), 1 ExternalModel, 1 MaaSModelRef, 1 MaaSSubscription, 1 MaaSAuthPolicy, plus the DSC patch.

Note: `oc kustomize` may warn about unknown CRD types (`maas.opendatahub.io/v1alpha1`). This is expected — the CRDs are installed by the RHOAI operator on the cluster, not in the local kustomize tree.

- [ ] **Step 10: Commit**

```bash
git add platform/base/models/
git commit -m "feat: add mock LM model registration with MaaS"
```

---

### Task 6: Kustomize overlays — dev, dev-gpu, demo

**Files:**

- Create: `platform/overlays/dev/kustomization.yaml`
- Create: `platform/overlays/dev-gpu/kustomization.yaml`
- Create: `platform/overlays/demo/kustomization.yaml`

- [ ] **Step 1: Create directories**

```bash
mkdir -p platform/overlays/{dev,dev-gpu,demo}
```

- [ ] **Step 2: Create platform/overlays/dev/kustomization.yaml**

The dev overlay uses the base as-is — mock models, minimal resources. This is the starting point for any new developer.

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - ../../base
```

- [ ] **Step 3: Create platform/overlays/dev-gpu/kustomization.yaml**

The dev-gpu overlay starts from the same base. When real models are added (future tasks), they will be added as additional resources here, and the mocklm deployment can be removed or kept alongside.

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - ../../base
```

- [ ] **Step 4: Create platform/overlays/demo/kustomization.yaml**

The demo overlay also starts from the base. Future tasks will add the full model set and resource scaling patches.

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - ../../base
```

- [ ] **Step 5: Verify all overlays render identically to base**

```bash
oc kustomize platform/overlays/dev
oc kustomize platform/overlays/dev-gpu
oc kustomize platform/overlays/demo
```

Expected: all three produce the same output as `oc kustomize platform/base/`. This is correct — the overlays don't diverge yet.

- [ ] **Step 6: Commit**

```bash
git add platform/overlays/
git commit -m "feat: add dev, dev-gpu, and demo kustomize overlays"
```

---

### Task 7: End-to-end validation on a live cluster

This task requires a cluster with all prerequisites installed (RHOAI 3.4+, RHCL 1.3+, cert-manager, default StorageClass).

**Files:**

- No file changes — execution and validation only

- [ ] **Step 1: Run preflight**

```bash
./platform/preflight.sh
```

Expected: all checks pass. If any fail, install the missing prerequisite per `platform/prerequisites.md` before continuing.

- [ ] **Step 2: Deploy dev overlay**

```bash
oc apply -k platform/overlays/dev
```

Expected output includes: namespaces created, DSC patched, PostgreSQL StatefulSet created, gateway created, mocklm deployment created, MaaS CRs created.

- [ ] **Step 3: Wait for pods**

```bash
oc wait --for=condition=Ready pod -l app=maas-db -n physical-ai --timeout=120s
oc wait --for=condition=Ready pod -l app=mocklm -n physical-ai-models --timeout=120s
```

- [ ] **Step 4: Verify mock LM responds directly**

```bash
oc run curl-test --rm -it --restart=Never \
  --image=registry.access.redhat.com/ubi9/ubi-minimal:latest \
  -- curl -s http://mocklm.physical-ai-models.svc:8080/v1/models
```

Expected: JSON response listing model `physical-ai-lm`.

- [ ] **Step 5: Verify MaaS resources reconcile**

```bash
oc get externalmodel -n physical-ai-models
oc get maasmodelref -n physical-ai-models
oc get maassubscription -n models-as-a-service
oc get maasauthpolicy -n models-as-a-service
```

Expected: all resources exist. Check status conditions for any errors — these will reveal if the MaaS controller is running and reconciling.

- [ ] **Step 6: Test MaaS gateway (if MaaS controller has reconciled)**

```bash
MAAS_URL="https://maas.$(oc get ingresses.config.openshift.io cluster -o jsonpath='{.spec.domain}')"
echo "MaaS URL: $MAAS_URL"
curl -sk "$MAAS_URL/health"
```

Expected: `{"status":"ok"}` from the MaaS API. If the gateway is not yet ready, the MaaS controller may still be provisioning — check `oc get pods -n redhat-ods-applications | grep maas`.

- [ ] **Step 7: Clean up**

```bash
oc delete -k platform/overlays/dev
```

Expected: all platform resources removed. Note: the DSC patch is a merge patch on an existing resource — `oc delete -k` will attempt to delete the DSC itself, which may not be desired. If this is a problem, the DSC patch should be applied/reverted separately. Document this as a known limitation if encountered.
