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
