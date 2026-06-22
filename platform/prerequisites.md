# Prerequisites

The Physical AI Platform requires the following operators and cluster
capabilities before deployment. The platform does not install these —
they must be set up by a cluster admin.

## Required Operators

| Operator | Min Version | OperatorHub Package | Purpose |
|----------|-------------|---------------------|---------|
| OpenShift Container Platform | 4.19.9 | — | Gateway API CRDs (native in 4.19+) |
| Red Hat OpenShift AI | 3.4 | `rhods-operator` | KServe, MaaS controller, Dashboard |
| Red Hat Connectivity Link | 1.3 | `rhcl-operator` | Kuadrant, Authorino, Limitador |
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

### Red Hat Connectivity Link

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

After the operator is ready, create the Kuadrant CR:

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
