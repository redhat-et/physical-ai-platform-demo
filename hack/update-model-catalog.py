# /// script
# requires-python = ">=3.11"
# dependencies = ["ruamel.yaml==0.18.16"]
# ///

"""Refresh allowlisted Hugging Face cards in the GitOps model catalog."""

import argparse
from collections.abc import Mapping
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from urllib.parse import quote, urljoin, urlsplit
from urllib.request import urlopen

from ruamel.yaml import YAML, YAMLError
from ruamel.yaml.comments import CommentedMap
from ruamel.yaml.scalarstring import LiteralScalarString


HERE = Path(__file__).resolve().parent
HUB = "https://huggingface.co"
SOURCE = "Physical AI"
REPO_PROPERTY = "physical_ai_hf_repo"
REVISION_PROPERTY = "physical_ai_hf_revision"
CONFIGMAP_LIMIT = 1024 * 1024
REPO_ID = re.compile(r"[\w.-]+/[\w.-]+", re.ASCII)
COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
FRONT_MATTER = re.compile(r"\A---[ \t]*\r?\n(.*?)^---[ \t]*(?:\r?\n|\Z)", re.MULTILINE | re.DOTALL)
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
INLINE_LINK = re.compile(r"(!?\[[^\]\n]*\]\(\s*)(<[^>\n]+>|[^\s)]+)")
REFERENCE_LINK = re.compile(r"^( {0,3}\[[^\]\n]+\]:\s+)(<[^>\n]+>|\S+)")
HTML_URL = re.compile(r"(\b(?:src|href|poster)\s*=\s*)([\"'])(.*?)\2", re.IGNORECASE)
OVERRIDABLE_STRINGS = {"description", "provider", "logo", "license", "licenseLink", "libraryName", "maturity"}
OVERRIDABLE_LISTS = {"language", "tasks"}


def required_text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value.strip()


def text_list(value):
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(item.strip() for item in value if isinstance(item, str) and item.strip()))


def read_allowlist(path):
    document = YAML(typ="safe").load(path)
    if not isinstance(document, dict) or set(document) != {"models"} or not isinstance(document["models"], list):
        raise ValueError("allowlist must contain a models list")
    entries = {}
    for entry in document["models"]:
        if not isinstance(entry, dict) or entry.keys() - {"name", "repo", "revision", "overrides", "supplement"}:
            raise ValueError("allowlist model must contain name, repo, and optional revision, overrides, supplement")
        name = required_text(entry.get("name"), "name")
        repo = required_text(entry.get("repo"), f"{name}.repo")
        if not REPO_ID.fullmatch(repo) or {".", ".."}.intersection(repo.split("/")):
            raise ValueError(f"{name}.repo must be a Hugging Face owner/repository ID")
        if name in entries:
            raise ValueError(f"duplicate allowlist name: {name}")
        ref = required_text(entry.get("revision", "main"), f"{name}.revision")
        supplement = entry.get("supplement", "")
        overrides = entry.get("overrides", {})
        if not isinstance(supplement, str) or not isinstance(overrides, dict):
            raise ValueError(f"{name}: supplement must be Markdown text and overrides must be a mapping")
        for key, value in overrides.items():
            if key in OVERRIDABLE_STRINGS:
                required_text(value, f"{name}.overrides.{key}")
            elif key == "customProperties":
                if not isinstance(value, dict):
                    raise ValueError(f"{name}.overrides.customProperties must map names to strings")
                for property_name, property_value in value.items():
                    required_text(property_name, f"{name}.customProperties key")
                    required_text(property_value, f"{name}.customProperties.{property_name}")
                    if property_name in {REPO_PROPERTY, REVISION_PROPERTY}:
                        raise ValueError(f"{name}: cannot override {property_name}")
            elif key not in OVERRIDABLE_LISTS or not isinstance(value, list) or len(text_list(value)) != len(value):
                raise ValueError(f"{name}: unsupported override or invalid value: {key}")
        entries[name] = (repo, ref, overrides, supplement)
    return entries


def download(opener, url):
    with opener(url, timeout=30) as response:
        data = response.read(CONFIGMAP_LIMIT + 1)
    if len(data) > CONFIGMAP_LIMIT:
        raise ValueError(f"response exceeds the catalog size limit: {url}")
    return data.decode("utf-8")


def absolute_link(target, repo, sha, kind="blob"):
    wrapped = target.startswith("<") and target.endswith(">")
    link = target[1:-1] if wrapped else target
    parts = urlsplit(link)
    if parts.scheme or parts.netloc or link.startswith("#") or not parts.path:
        return target
    if link.startswith("/"):
        absolute = urljoin(HUB, link)
    else:
        absolute = urljoin(f"{HUB}/{repo}/{kind}/{sha}/", quote(link, safe="%/?:@&=+#,;~!$'()*[]"))
    return f"<{absolute}>" if wrapped else absolute


def pin_readme_links(markdown, repo, sha):
    def rewrite(line):
        line = INLINE_LINK.sub(
            lambda m: m[1] + absolute_link(m[2], repo, sha, "resolve" if m[1].startswith("!") else "blob"),
            line,
        )
        # Reference definitions can also be used by images, so use the raw asset URL.
        line = REFERENCE_LINK.sub(lambda m: m[1] + absolute_link(m[2], repo, sha, "resolve"), line)
        return HTML_URL.sub(
            lambda m: m[1] + m[2] + absolute_link(m[3], repo, sha, "blob" if m[1].lower().startswith("href") else "resolve") + m[2],
            line,
        )

    lines = []
    fence = None
    for line in markdown.splitlines(keepends=True):
        match = FENCE.match(line)
        if match and fence is None:
            fence = match[1]
        elif match and match[1].startswith(fence) and not match[2].strip():
            fence = None
        elif fence is None:
            # Keep inline code literal, just like fenced examples.
            parts = re.split(r"(`+[^`]+`+)", line)
            line = "".join(rewrite(part) if i % 2 == 0 else part for i, part in enumerate(parts))
            # CommonMark accepts backslashes for hard breaks without trailing blanks in the YAML.
            if line.endswith("  \n") and line.strip() and not line.startswith((" ", "\t")):
                line = line.rstrip() + "\\\n"
        lines.append(line.rstrip(" \t\n") + ("\n" if line.endswith("\n") else ""))
    return "".join(lines)


def metadata_string(value):
    return {"metadataType": "MetadataStringValue", "string_value": value}


def model_from_hub(opener, name, repo, ref, overrides, supplement):
    info = json.loads(download(opener, f"{HUB}/api/models/{repo}/revision/{quote(ref, safe='')}"))
    if not isinstance(info, dict) or info.get("private") or info.get("gated"):
        raise ValueError("only public, non-gated HF models are supported")
    sha = required_text(info.get("sha"), "Hub commit SHA")
    if not COMMIT_SHA.fullmatch(sha) or (COMMIT_SHA.fullmatch(ref) and sha != ref):
        raise ValueError("HF returned an invalid or unexpected commit SHA")
    raw = download(opener, f"{HUB}/{repo}/raw/{sha}/README.md")
    header = FRONT_MATTER.match(raw)
    front_matter = YAML(typ="safe").load(header[1]) if header else None
    if front_matter is None:
        front_matter = {}
    if not isinstance(front_matter, dict) or (raw.splitlines()[:1] == ["---"] and not header):
        raise ValueError("HF README has invalid YAML front matter")
    markdown = (raw[header.end():] if header else raw).strip("\r\n")
    required_text(markdown, "HF README body")
    hub_metadata = info.get("cardData") or {}
    if not isinstance(hub_metadata, dict):
        raise ValueError("HF cardData must be a mapping")
    card = {**hub_metadata, **front_matter}
    license_name = card.get("license")
    if license_name == "other":
        license_name = card.get("license_name") or license_name
    license_link = card.get("license_link")
    fields = {
        "name": name,
        "description": card.get("description") or info.get("description"),
        "provider": info.get("author") or repo.split("/", 1)[0],
        "license": license_name,
        "licenseLink": absolute_link(license_link, repo, sha) if isinstance(license_link, str) else None,
        "libraryName": info.get("library_name") or card.get("library_name"),
        "language": text_list(card.get("language")),
        "tasks": text_list(info.get("pipeline_tag") or card.get("pipeline_tag")),
    }
    fields = {key: value for key, value in fields.items() if value and isinstance(value, (str, list))}
    fields.update((key, value) for key, value in overrides.items() if key != "customProperties")
    card_url = f"{HUB}/{repo}/blob/{sha}/README.md"
    snapshot_url = f"{HUB}/{repo}/tree/{sha}"
    readme = pin_readme_links(markdown, repo, sha) + (
        f"\n\n[Hugging Face snapshot at `{sha[:12]}`]({snapshot_url})"
        f" · [Upstream model card]({card_url})"
    )
    if supplement.strip():
        readme += "\n\n## Physical AI notes\n\n" + supplement.strip()
    fields["readme"] = LiteralScalarString(readme)
    fields["customProperties"] = {REPO_PROPERTY: metadata_string(repo), REVISION_PROPERTY: metadata_string(sha)}
    fields["customProperties"].update(
        (key, metadata_string(value)) for key, value in sorted(overrides.get("customProperties", {}).items())
    )
    # The RHOAI 3.5 static YAML importer rejects hf:// artifact URIs.
    # Keep the pinned Hub snapshot as card provenance, not an artifact.
    return fields


def refresh(allowlist, catalog, opener, check):
    entries = read_allowlist(allowlist)
    original = catalog.read_bytes()
    yaml = YAML()
    yaml.preserve_quotes = True
    # Don't wrap the existing anchored MockLM logo or other single-line scalars.
    yaml.width = CONFIGMAP_LIMIT
    doc = yaml.load(original.decode("utf-8"))
    if not isinstance(doc, Mapping) or doc.get("source") != SOURCE or not isinstance(doc.get("models"), list):
        raise ValueError(f"{catalog} must contain the {SOURCE!r} source and its models list")

    existing = {}
    owned = set()
    for model in doc["models"]:
        if not isinstance(model, Mapping):
            raise ValueError("catalog model must be a mapping")
        name = required_text(model.get("name"), "catalog model name")
        if name in existing:
            raise ValueError(f"duplicate existing catalog name: {name}")
        existing[name] = model
        properties = model.get("customProperties") or {}
        if not isinstance(properties, Mapping):
            raise ValueError(f"{name}: catalog customProperties must be a mapping")
        marker = properties.get(REPO_PROPERTY)
        if isinstance(marker, Mapping) and marker.get("metadataType") == "MetadataStringValue" and marker.get("string_value"):
            owned.add(name)
        elif name in entries:
            raise ValueError(f"allowlist name {name!r} collides with an unmanaged catalog entry")

    generated = {}
    for name, (repo, ref, overrides, supplement) in entries.items():
        try:
            generated[name] = model_from_hub(opener, name, repo, ref, overrides, supplement)
        except (OSError, ValueError, YAMLError) as exc:
            raise ValueError(f"{name} ({repo}): {exc}") from exc
        previous = existing.get(name)
        if previous and previous.get("license") != generated[name].get("license"):
            print(f"Review license change for {name}: {previous.get('license')!r} → {generated[name].get('license')!r}", file=sys.stderr)

    # Keep the round-trip nodes for unmanaged entries (including their comments and aliases).
    models = doc["models"]
    for index in range(len(models) - 1, -1, -1):
        if models[index]["name"] in owned:
            del models[index]
    models.extend(CommentedMap(model) for model in generated.values())
    for name in sorted(owned - entries.keys()):
        print(f"Remove formerly generated entry: {name}")
    buffer = io.StringIO()
    yaml.dump(doc, buffer)
    output = buffer.getvalue().encode("utf-8")
    sources = catalog.with_name("sources.yaml")
    if len(output) + (sources.stat().st_size if sources.is_file() else 0) > CONFIGMAP_LIMIT:
        raise ValueError("catalog data exceeds the Kubernetes ConfigMap 1 MiB limit")

    if output == original:
        print(f"Catalog is up to date ({len(generated)} HF entries).")
        return 0
    if check:
        print("Catalog refresh available; run without --check and review the diff.")
        return 1
    if catalog.read_bytes() != original:
        raise ValueError("catalog changed during refresh; refusing to overwrite it")
    mode = catalog.stat().st_mode & 0o777
    with tempfile.NamedTemporaryFile(dir=catalog.parent, prefix=".catalog-", delete=False) as temporary:
        try:
            temporary.write(output)
            temporary.close()
            os.chmod(temporary.name, mode)
            os.replace(temporary.name, catalog)
        finally:
            Path(temporary.name).unlink(missing_ok=True)
    print(f"Refreshed {len(generated)} HF entries. Review the catalog diff before committing.")
    return 0


def main(argv=None, *, opener=urlopen):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allowlist", type=Path, default=HERE / "model-catalog.yaml")
    parser.add_argument("--catalog", type=Path, default=HERE.parent / "platform/base/model-catalog/catalog.yaml")
    parser.add_argument("--check", action="store_true", help="exit 1 if a refresh would change the catalog; do not write")
    args = parser.parse_args(argv)
    try:
        return refresh(args.allowlist, args.catalog, opener, args.check)
    except (OSError, ValueError, YAMLError) as exc:
        print(f"Unable to refresh catalog: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
