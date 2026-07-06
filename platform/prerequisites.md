# Prerequisites

The Physical AI Platform requires the following operators and cluster
capabilities before deployment. The platform does not install these —
they must be set up by a cluster admin.

## Required Operators

| Operator | Min Version | OperatorHub Package | Catalog | Purpose |
| ---------- | ------------- | --------------------- | --------- | --------- |
| OpenShift Container Platform | 4.19.9 | — | — | Gateway API CRDs (native in 4.19+) |
| Red Hat OpenShift AI | 3.4 | `rhods-operator` | `redhat-operators` | KServe, MaaS controller, Dashboard |
| Red Hat Connectivity Link | 1.3 | `rhcl-operator` | `redhat-operators` | Kuadrant, Authorino, Limitador |
| cert-manager | 1.x | `openshift-cert-manager-operator` | `redhat-operators` | TLS certificates |
| Custom Metrics Autoscaler | 2.19 | `custom-metrics-autoscaler` | `redhat-operators` | KEDA for GPU model scale-to-zero (optional) |

> **Note:** Do not install the community `kuadrant-operator` from
> `community-operators`. It is deprecated and its CRDs are incompatible
> with RHOAI 3.4's MaaS controller. Use `rhcl-operator` from
> `redhat-operators` instead.

## Required Cluster Configuration

- **Default StorageClass** — any CSI provisioner with a default StorageClass.
  Verify with: `oc get sc`
- **User Workload Monitoring** — must be enabled for MaaS metrics.

## Installation Guidance

### Red Hat OpenShift AI

Install from OperatorHub (Installed Operators → Red Hat OpenShift AI) or via
CLI. The operator requires its own namespace with an OperatorGroup:

```bash
oc create namespace redhat-ods-operator 2>/dev/null || true
cat <<'EOF' | oc apply -f -
apiVersion: operators.coreos.com/v1
kind: OperatorGroup
metadata:
  name: redhat-ods-operator
  namespace: redhat-ods-operator
spec: {}
---
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

Wait for the CSV to reach `Succeeded`:

```bash
oc get csv -n redhat-ods-operator -w
```

The operator auto-creates a `DSCInitialization` CR. The `DataScienceCluster`
CR is created by the platform's kustomize manifests — do **not** create one
manually.

### Red Hat Connectivity Link

The RHCL operator installs into its own namespace (`kuadrant-system`) with
its dependency operators (Authorino, Limitador, DNS):

```bash
oc create namespace kuadrant-system 2>/dev/null || true
cat <<'EOF' | oc apply -f -
apiVersion: operators.coreos.com/v1
kind: OperatorGroup
metadata:
  name: kuadrant
  namespace: kuadrant-system
spec:
  upgradeStrategy: Default
---
apiVersion: operators.coreos.com/v1alpha1
kind: Subscription
metadata:
  name: rhcl-operator
  namespace: kuadrant-system
spec:
  channel: stable
  name: rhcl-operator
  source: redhat-operators
  sourceNamespace: openshift-marketplace
  installPlanApproval: Automatic
EOF
```

Wait for the CSV to reach `Succeeded`:

```bash
oc get csv -n kuadrant-system -w
```

Then create the Kuadrant CR:

```bash
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

## Database Secrets

The platform requires PostgreSQL database secrets that are **not** managed by
ArgoCD. These must be provisioned before deployment.

| Secret | Namespace | Purpose |
| -------- | ----------- | --------- |
| `maas-db-admin-credentials` | `physical-ai` | PostgreSQL admin password for MaaS DB |
| `maas-db-credentials` | `physical-ai` | MaaS DB user, password, database name |
| `maas-db-config` | `redhat-ods-applications` | MaaS DB connection URL |
| `mlflow-db-admin-credentials` | `physical-ai` | PostgreSQL admin password for MLflow DB |
| `mlflow-db-credentials` | `physical-ai` | MLflow DB user, password, database name |
| `mlflow-db-connection` | `redhat-ods-applications` | MLflow DB connection URL |

Use `--generate-secrets` to create them with random passwords:

```bash
./platform/preflight.sh --generate-secrets
```

The flag is idempotent — it skips secrets that already exist and reads back
existing passwords when generating dependent secrets (e.g. connection URLs).

## Validation

Run the preflight script to check all prerequisites:

```bash
./platform/preflight.sh
```

## Optional: Custom Metrics Autoscaler (KEDA)

Required for GPU model scale-to-zero. Without it, GPU models stay running
continuously. With it, models like Qwen3-Omni and Cosmos3-Nano scale to zero
when idle and wake up on demand through MaaS requests.

### Install the operator

Install **Custom Metrics Autoscaler** from the Software Catalog (search
"Custom Metrics Autoscaler"). It installs into `openshift-keda`.

### Create the KedaController with HTTP Add-on

```bash
oc patch kedacontroller keda -n openshift-keda --type=merge \
  -p '{"spec":{"httpAddon":{"enabled":true}}}'
```

The HTTP Add-on deploys an interceptor proxy that buffers requests while
scaled-to-zero pods start up.

### Set up Prometheus RBAC

KEDA needs to read Prometheus metrics for scaling decisions:

```bash
# Service account and token
cat <<'EOF' | oc apply -f -
apiVersion: v1
kind: ServiceAccount
metadata:
  name: keda-prometheus-sa
  namespace: openshift-keda
---
apiVersion: v1
kind: Secret
metadata:
  name: keda-prometheus-token
  namespace: openshift-keda
  annotations:
    kubernetes.io/service-account.name: keda-prometheus-sa
type: kubernetes.io/service-account-token
EOF

# Grant cluster monitoring read access
oc adm policy add-cluster-role-to-user cluster-monitoring-view \
  -z keda-prometheus-sa -n openshift-keda
```

### Create TriggerAuthentication

Copy the token to the model namespace and create the TriggerAuthentication:

```bash
TOKEN=$(oc get secret keda-prometheus-token -n openshift-keda -o jsonpath='{.data.token}')
CA=$(oc get secret keda-prometheus-token -n openshift-keda -o jsonpath='{.data.ca\.crt}')

cat <<EOF | oc apply -f -
apiVersion: v1
kind: Secret
metadata:
  name: keda-prometheus-token
  namespace: physical-ai-models
type: Opaque
data:
  token: $TOKEN
  ca.crt: $CA
---
apiVersion: keda.sh/v1alpha1
kind: TriggerAuthentication
metadata:
  name: keda-trigger-auth-prometheus
  namespace: physical-ai-models
spec:
  secretTargetRef:
  - parameter: bearerToken
    name: keda-prometheus-token
    key: token
  - parameter: ca
    name: keda-prometheus-token
    key: ca.crt
EOF
```

### How it works

GPU models (Qwen3-Omni, Cosmos3-Nano, DreamZero) include an
`HTTPScaledObject` in their kustomization that tells KEDA to manage
scaling. The MaaS proxy routes these models through the KEDA HTTP
interceptor, which buffers requests during cold starts. Models scale to
zero after 1 hour of no traffic and wake up when a MaaS request arrives.

The first request after scale-to-zero may take several minutes while the
model loads into GPU memory. Use a long timeout (e.g. `curl -m 900`).
