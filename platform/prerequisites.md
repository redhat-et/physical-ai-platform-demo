# Prerequisites

The Physical AI Platform requires the following operators and cluster
capabilities before deployment. The platform does not install these —
they must be set up by a cluster admin.

## Required Operators

| Operator | Min Version | OperatorHub Package | Catalog | Purpose |
|----------|-------------|---------------------|---------|---------|
| OpenShift Container Platform | 4.19.9 | — | — | Gateway API CRDs (native in 4.19+) |
| Red Hat OpenShift AI | 3.4 | `rhods-operator` | `redhat-operators` | KServe, MaaS controller, Dashboard |
| Red Hat Connectivity Link | 1.3 | `rhcl-operator` | `redhat-operators` | Kuadrant, Authorino, Limitador |
| cert-manager | 1.x | `openshift-cert-manager-operator` | `redhat-operators` | TLS certificates |

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

## Validation

Run the preflight script to check all prerequisites:

```bash
./platform/preflight.sh
```
