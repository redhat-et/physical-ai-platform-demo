# ArgoCD GitOps Deployment

Deploy the Physical AI Platform using GitOps with ArgoCD. Changes pushed to Git automatically sync to your cluster.

## Quick Start

```bash
# 1. Apply the ArgoCD Application
oc apply -f argocd/application-dev-gpu.yaml

# 2. Grant ArgoCD permissions
oc apply -f argocd/permissions.yaml

# 3. Watch deployment
oc get application physical-ai-platform-dev-gpu -n openshift-gitops -w
```

That's it! ArgoCD syncs from GitHub and deploys the platform.

## Prerequisites

- **OpenShift cluster** with OpenShift GitOps operator installed
- **Cluster admin access** to create RoleBindings
- **GPU nodes** (for dev-gpu profile)

## Applications

Choose one based on your environment:

| Application | Profile | GPU | Use Case |
|-------------|---------|-----|----------|
| `application-dev.yaml` | Standard Dev | No | Testing without GPU |
| `application-dev-gpu.yaml` | GPU Dev | Yes | GPU workloads like DreamZero |
| `application-demo.yaml` | Full Demo | Yes | Production-like demo |

All applications deploy from `platform/overlays/<profile>` which currently reference the same base.

## Access ArgoCD UI

**Get URL:**
```bash
oc get route openshift-gitops-server -n openshift-gitops -o jsonpath='{.spec.host}'
```

**Get password:**
```bash
oc get secret openshift-gitops-cluster -n openshift-gitops -o jsonpath='{.data.admin\.password}' | base64 -d && echo
```

**Login:**
- Username: `admin`
- Password: (from command above)

## GitOps Workflow

Once deployed, your workflow becomes:

```bash
# 1. Edit platform manifests
vim platform/base/models/mocklm/deployment.yaml

# 2. Commit and push
git add platform/base/models/mocklm/deployment.yaml
git commit -m "Scale mocklm to 2 replicas"
git push

# 3. ArgoCD auto-syncs within ~3 minutes
# Or force immediate sync from the UI
```

### Sync Behavior

All applications are configured with:

- ✅ **Auto-sync**: Git changes automatically deploy
- ✅ **Self-heal**: Manual cluster changes revert to Git state
- ✅ **Prune**: Resources removed from Git are deleted from cluster

**Example:** If you manually run `oc scale deployment mocklm --replicas=3`, ArgoCD will revert it back to what's in Git within minutes.

## Monitoring Deployment

**Check sync status:**
```bash
oc get application -n openshift-gitops
```

**View details:**
```bash
oc describe application physical-ai-platform-dev-gpu -n openshift-gitops
```

**Watch deployed resources:**
```bash
oc get all -n physical-ai
oc get all -n physical-ai-models
```

**Check in ArgoCD UI:**
- Navigate to the application
- View resource tree
- See sync status and health

## Troubleshooting

### Application Shows OutOfSync

**Force sync:**
```bash
oc patch application physical-ai-platform-dev-gpu -n openshift-gitops --type merge -p '{"operation": {"sync": {}}}'
```

**Or sync from UI:** Click "Sync" → "Synchronize"

### Permission Errors

If you see errors like `User "system:serviceaccount:openshift-gitops:..." cannot patch resource`:

```bash
# Reapply permissions
oc apply -f argocd/permissions.yaml
```

### Application Stuck

**Delete and recreate:**
```bash
oc delete application physical-ai-platform-dev-gpu -n openshift-gitops
oc apply -f argocd/application-dev-gpu.yaml
```

## Deploying Models via GitOps

To deploy DreamZero through ArgoCD:

1. **Ensure ServingRuntime exists:**
   ```bash
   oc apply -f platform/base/models/dreamzero/servingruntime-fixed.yaml
   ```

2. **Add InferenceService to kustomization** (if not already there)

3. **Commit and push:**
   ```bash
   git add platform/base/models/dreamzero/
   git commit -m "Add DreamZero model deployment"
   git push
   ```

4. **ArgoCD auto-syncs** and deploys DreamZero

See `platform/base/models/dreamzero/README.md` for DreamZero-specific details.
