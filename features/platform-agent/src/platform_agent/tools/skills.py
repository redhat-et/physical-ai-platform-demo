from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import yaml
from langchain_core.tools import tool

_SKILLS_PACKAGE = "platform_agent.skills"
_FRONTMATTER_RE = re.compile(r"\A---\n(.*?\n)---\n(.*)\Z", re.DOTALL)


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str


def _parse_skill_file(text: str, filename: str) -> Skill:
    """Parse a skill .md file's leading '---'-delimited YAML frontmatter
    (name, description) plus its markdown body. Takes raw text rather than
    a path so it's directly unit-testable with string literals.
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        raise ValueError(f"{filename}: missing leading '---' YAML frontmatter block")
    frontmatter_raw, body = match.groups()
    meta = yaml.safe_load(frontmatter_raw) or {}
    for field in ("name", "description"):
        if not meta.get(field):
            raise ValueError(f"{filename}: frontmatter missing required '{field}' field")
    return Skill(name=meta["name"], description=meta["description"], body=body.strip())


@lru_cache(maxsize=1)
def _load_skills() -> dict[str, Skill]:
    skills: dict[str, Skill] = {}
    for entry in sorted(resources.files(_SKILLS_PACKAGE).iterdir(), key=lambda p: p.name):
        if entry.suffix != ".md":
            continue
        skill = _parse_skill_file(entry.read_text(encoding="utf-8"), entry.name)
        if skill.name in skills:
            raise ValueError(f"duplicate skill name '{skill.name}' (in {entry.name})")
        skills[skill.name] = skill
    return skills


def skills_index() -> str:
    """Render the always-on index for the system prompt: one 'name:
    description' line per skill, sorted by name for determinism.
    """
    return "\n".join(f"- {s.name}: {s.description}" for s in _load_skills().values())


@tool
def get_skill(name: str) -> str:
    """Load the full step-by-step procedure for one of the named workflow
    skills advertised in the SKILLS section of your instructions. Call this
    before your first tool call in a matching workflow — the one-line
    description alone is not enough to act correctly.

    Args:
        name: The exact skill name from the SKILLS section (e.g. 'datasets').
    """
    skills = _load_skills()
    skill = skills.get(name)
    if skill is None:
        return f"No skill named '{name}'. Available: {sorted(skills.keys())}."
    return skill.body
