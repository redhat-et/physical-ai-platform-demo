# MaaS Infrastructure

## Cluster Prerequisites

These one-time steps configure Authorino TLS per [RHOAI docs Section 1.4](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/govern_llm_access_with_models-as-a-service/deploy-and-manage-models-as-a-service#configure-tls-for-models-as-a-service_govern-llm-access-with-models-as-a-service). They are applied to operator-managed resources and may need to be re-applied after operator upgrades.

```bash
# 1. Generate TLS cert for Authorino
oc annotate service authorino-authorino-authorization -n kuadrant-system \
  service.beta.openshift.io/serving-cert-secret-name=authorino-server-cert --overwrite

# 2. Enable Authorino TLS listener
oc patch authorino authorino -n kuadrant-system --type=merge --patch '
{
  "spec": {
    "listener": {
      "tls": {
        "enabled": true,
        "certSecretRef": { "name": "authorino-server-cert" }
      }
    }
  }
}'

# 3. Configure Authorino to trust the OpenShift service CA
oc -n kuadrant-system set env deployment/authorino \
  SSL_CERT_FILE=/etc/ssl/certs/openshift-service-ca/service-ca-bundle.crt \
  REQUESTS_CA_BUNDLE=/etc/ssl/certs/openshift-service-ca/service-ca-bundle.crt

# 4. Enable TLS bootstrap on the MaaS gateway
oc annotate gateway maas-default-gateway -n openshift-ingress \
  security.opendatahub.io/authorino-tls-bootstrap="true" --overwrite
```

### Verification

```bash
oc get authorino authorino -n kuadrant-system -o jsonpath='{.spec.listener.tls.enabled}'
# Expected: true

oc get deployment/authorino -n kuadrant-system -o \
  jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="SSL_CERT_FILE")].value}'
# Expected: /etc/ssl/certs/openshift-service-ca/service-ca-bundle.crt

oc get gateway maas-default-gateway -n openshift-ingress -o \
  jsonpath='{.metadata.annotations.security\.opendatahub\.io/authorino-tls-bootstrap}'
# Expected: true
```

## API Keys

Generate an API key to access models through MaaS. Keys are scoped to a subscription which determines which models the key can access and the rate limits.

**Via CLI** (from inside the cluster or with port-forward to `maas-api`):

```bash
curl -sk https://maas-api.redhat-ods-applications.svc:8443/v1/api-keys \
  -X POST \
  -H "X-MaaS-Username: $(oc whoami)" \
  -H 'X-MaaS-Group: ["system:authenticated"]' \
  -H "Content-Type: application/json" \
  -d '{"name":"my-key","subscription":"physical-ai-dev"}'
```

**Via dashboard**: Gen AI studio → API keys → Create API key.

The response contains the `key` field (prefixed `sk-oai-`). Save it — it cannot be retrieved later.

## Testing

From inside the cluster, use the API key to call a model through the MaaS gateway:

```bash
curl -s http://maas-default-gateway-data-science-gateway-class.openshift-ingress.svc.cluster.local/physical-ai-models/mocklm-echo/v1/chat/completions \
  -H "Authorization: Bearer <your-sk-oai-key>" \
  -H "Content-Type: application/json" \
  -d '{"model":"mocklm-echo","messages":[{"role":"user","content":"Hello"}]}'
```