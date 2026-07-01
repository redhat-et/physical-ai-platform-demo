#!/usr/bin/env bash
set -euo pipefail

PASS=0
FAIL=0
WARN=0
GENERATE_SECRETS=false

check_pass() { echo "  ✓ $*"; PASS=$((PASS + 1)); }
check_fail() { echo "  ✗ $*"; FAIL=$((FAIL + 1)); }
check_warn() { echo "  ! $*"; WARN=$((WARN + 1)); }

usage() {
    echo "Usage: $0 [--generate-secrets]"
    echo ""
    echo "Options:"
    echo "  --generate-secrets  Generate required database secrets if not present"
    exit 0
}

for arg in "$@"; do
    case "$arg" in
        --generate-secrets) GENERATE_SECRETS=true ;;
        --help|-h) usage ;;
        *) echo "Unknown option: $arg"; usage ;;
    esac
done

generate_password() {
    openssl rand -base64 18 | tr -d '/+=' | head -c 24
}

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

check_operator() {
    local label="$1"
    local sub_ns="$2"
    local sub_name="$3"
    local min_version="$4"
    local csv version
    csv="$(oc get subscription "$sub_name" -n "$sub_ns" -o jsonpath='{.status.currentCSV}' 2>/dev/null || true)"
    version="${csv#"${sub_name}."}"
    version="${version#v}"
    if [[ -n "$version" ]]; then
        if awk "BEGIN{exit !(\"$version\" >= \"$min_version\")}"; then
            check_pass "$label $version (>= $min_version required)"
        else
            check_fail "$label $version found, but >= $min_version required"
        fi
    else
        check_fail "$label not installed (>= $min_version required) — see platform/prerequisites.md"
    fi
}

check_operator "Red Hat OpenShift AI" "redhat-ods-operator" "rhods-operator" "3.4"
check_operator "Red Hat Connectivity Link" "kuadrant-system" "rhcl-operator" "1.3"
check_operator "cert-manager Operator" "cert-manager-operator" "openshift-cert-manager-operator" "1.0"
echo ""

# --- Kuadrant CR ---
echo "Kuadrant:"
if oc get kuadrant -n kuadrant-system &>/dev/null; then
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

# --- Database secrets ---
echo "Database secrets:"

ensure_namespace() {
    local ns="$1"
    if ! oc get namespace "$ns" &>/dev/null; then
        if [[ "$GENERATE_SECRETS" == true ]]; then
            oc create namespace "$ns"
            echo "    Created namespace $ns"
        fi
    fi
}

check_secret() {
    local ns="$1"
    local name="$2"
    if oc get secret "$name" -n "$ns" &>/dev/null; then
        check_pass "Secret $name in $ns"
        return 0
    else
        if [[ "$GENERATE_SECRETS" == true ]]; then
            return 1
        else
            check_fail "Secret $name not found in $ns — run with --generate-secrets to create"
            return 1
        fi
    fi
}

generate_db_secrets() {
    local prefix="$1"
    local db_name="$2"
    local db_ns="$3"
    local conn_secret="$4"
    local conn_ns="$5"

    local password
    password="$(generate_password)"
    local admin_password
    admin_password="$(generate_password)"

    ensure_namespace "$db_ns"

    if ! oc get secret "${prefix}-db-admin-credentials" -n "$db_ns" &>/dev/null; then
        oc create secret generic "${prefix}-db-admin-credentials" \
            --namespace="$db_ns" \
            --from-literal=POSTGRESQL_ADMIN_PASSWORD="$admin_password"
        check_pass "Secret ${prefix}-db-admin-credentials created in $db_ns"
    fi

    if ! oc get secret "${prefix}-db-credentials" -n "$db_ns" &>/dev/null; then
        oc create secret generic "${prefix}-db-credentials" \
            --namespace="$db_ns" \
            --from-literal=POSTGRESQL_USER="$db_name" \
            --from-literal=POSTGRESQL_PASSWORD="$password" \
            --from-literal=POSTGRESQL_DATABASE="$db_name"
        check_pass "Secret ${prefix}-db-credentials created in $db_ns"
    else
        password="$(oc get secret "${prefix}-db-credentials" -n "$db_ns" -o jsonpath='{.data.POSTGRESQL_PASSWORD}' | base64 -d)"
    fi

    ensure_namespace "$conn_ns"

    if ! oc get secret "$conn_secret" -n "$conn_ns" &>/dev/null; then
        oc create secret generic "$conn_secret" \
            --namespace="$conn_ns" \
            --from-literal=DB_CONNECTION_URL="postgresql://${db_name}:${password}@${prefix}-db.${db_ns}.svc:5432/${db_name}?sslmode=disable"
        check_pass "Secret $conn_secret created in $conn_ns"
    fi
}

check_db_secrets() {
    local prefix="$1"
    local db_name="$2"
    local db_ns="$3"
    local conn_secret="$4"
    local conn_ns="$5"
    local needs_generate=false

    check_secret "$db_ns" "${prefix}-db-admin-credentials" || needs_generate=true
    check_secret "$db_ns" "${prefix}-db-credentials" || needs_generate=true
    check_secret "$conn_ns" "$conn_secret" || needs_generate=true

    if [[ "$needs_generate" == true && "$GENERATE_SECRETS" == true ]]; then
        generate_db_secrets "$prefix" "$db_name" "$db_ns" "$conn_secret" "$conn_ns"
    fi
}

#                  prefix    db_name  db_ns          conn_secret         conn_ns
check_db_secrets   "maas"    "maas"   "physical-ai"  "maas-db-config"    "redhat-ods-applications"
check_db_secrets   "mlflow"  "mlflow" "physical-ai"  "mlflow-db-connection" "physical-ai"
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
