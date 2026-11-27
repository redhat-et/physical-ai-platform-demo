# Agent identities

The agent touches the cluster through two separate ServiceAccounts in `physical-ai`.

| Identity | Held by | Access |
|----------|---------|--------|
| `pai-agent` | Claude Code and the platform agent (`oc`) | Read-only: `view` in `physical-ai`, `physical-ai-models`, `models-as-a-service`; nodes/namespaces; cluster monitoring. No Secrets, no write verbs. |
| `pai-mcp-write` | `pai-mcp-server` pod only | Reads plus the specific write verbs the MCP tools need (see `platform/base/pai-mcp/rbac.yaml`). The agent never sees this token. |
| `mcp-viewer` | `openshift-mcp-server` | Read-only, backs the `k8s_*` proxy tools. Unchanged. |

## Agent token

Mint a short-lived token and put it in a kubeconfig referenced from the user's Claude
`settings.json` `env`. Setup and verification: [README](../README.md#agent-access-claude-code--platform-agent).

## Rules for the MCP write surface

- RBAC bounds what the server can do; the typed tools bound what the agent may ask for.
- No generic `apply_manifest` / `k8s_resources_create_or_update`, and `run_script` must not
  run mutating scripts with the write SA unless each is exposed as its own typed tool.
- Mutations go through `plan_*` then `apply_*(plan_id, confirm=True)`, with plans stored
  server-side and bound to the rendered manifest.

## Open items

- The pod runs one SA, so `pai-mcp-write` still carries the union of read and write verbs.
  Audit the verbs in `pai-mcp/rbac.yaml` (e.g. `pods` delete, `inferenceservices` delete,
  `deployments/scale`) against the tools that actually need them and trim.
- Per-user attribution needs the OIDC path plus impersonation or token exchange.
