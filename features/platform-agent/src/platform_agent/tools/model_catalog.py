import httpx
from langchain_core.tools import tool

from platform_agent.config import settings


def _extract_section(text: str, section: str) -> tuple[str | None, list[str]]:
    """Pull out one '## <title>' block (verbatim, header included) from a
    markdown doc, matching `section` case-insensitively against exact H2
    titles. Returns (None, all_h2_titles) if no H2 in the doc matches, so
    the caller can tell the model what sections actually exist instead of
    just failing.
    """
    lines = text.splitlines()
    headers: list[str] = []
    start = None
    for i, line in enumerate(lines):
        if line.startswith("## "):
            title = line[3:].strip()
            headers.append(title)
            if start is None and title.lower() == section.strip().lower():
                start = i
    if start is None:
        return None, headers
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("## "):
            end = j
            break
    return "\n".join(lines[start:end]).rstrip(), headers


@tool
def get_model_readme(model_name: str, section: str | None = None) -> str:
    """Fetch a catalog model's README.md from this platform's own repo
    (platform/base/models/<model_name>/README.md) -- includes the model's
    Dataset Compatibility section (embodiment, action space, perceptual
    setup, etc.) needed to apply the datasets skill's checklist against a
    candidate dataset. model_name is a catalog directory name (e.g. 'pi05',
    'dreamzero'), not a Hugging Face dataset repo id.

    Args:
        model_name: Catalog model directory name under platform/base/models/,
            e.g. 'pi05' or 'dreamzero'.
        section: Optional exact '## <title>' heading to fetch instead of the
            whole file, e.g. 'Dataset Compatibility' -- avoids digesting
            unrelated Deployment/Testing/Troubleshooting content. Case
            -insensitive, but must otherwise match a real heading in the
            file; a near-miss returns the list of real headings rather than
            guessing which one you meant.
    """
    url = f"{settings.model_catalog_raw_base}/platform/base/models/{model_name}/README.md"
    try:
        resp = httpx.get(url, timeout=10.0)
    except httpx.HTTPError as e:
        return f"Could not fetch README for '{model_name}': {e}"
    if resp.status_code == 404:
        return (
            f"No README.md found for model '{model_name}' under "
            f"platform/base/models/. Check the model name."
        )
    resp.raise_for_status()
    text = resp.text

    if section is None:
        return f"platform/base/models/{model_name}/README.md:\n{text}"

    extracted, headers = _extract_section(text, section)
    if extracted is None:
        return (
            f"No '## {section}' section in platform/base/models/{model_name}/README.md. "
            f"Sections present: {headers}. Not every catalog model has a Dataset "
            f"Compatibility section -- only ones with a real fine-tuning recipe or "
            f"documented training data to check a candidate against."
        )
    return f"platform/base/models/{model_name}/README.md#{section}:\n{extracted}"
