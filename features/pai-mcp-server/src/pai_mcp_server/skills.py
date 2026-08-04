from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from pai_mcp_server.config import settings

_FRONTMATTER_RE = re.compile(r"\A---\n(.*?\n)---\n(.*)\Z", re.DOTALL)


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str
    dir_name: str


def _parse_skill_file(text: str, filename: str, dir_name: str) -> Skill:
    """Parses a SKILL.md file's leading '---'-delimited YAML frontmatter
    (name, description) plus its markdown body -- same contract as the
    frontmatter platform_agent's tools/skills.py already parses, kept in
    sync by convention since both read the same upstream skills repo.
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        raise ValueError(f"{filename}: missing leading '---' YAML frontmatter block")
    frontmatter_raw, body = match.groups()
    meta = yaml.safe_load(frontmatter_raw) or {}
    for field in ("name", "description"):
        if not meta.get(field):
            raise ValueError(f"{filename}: frontmatter missing required '{field}' field")
    return Skill(name=meta["name"], description=meta["description"], body=body.strip(), dir_name=dir_name)


def load_skills() -> dict[str, Skill]:
    """Re-reads settings.skills_root from disk on every call rather than
    caching -- a git-sync sidecar keeps that path current independently of
    this process's lifetime, and the point of that design is for a skill
    update to become visible without even restarting this server, let
    alone rebuilding it. The read itself is cheap (a handful of markdown
    files), so there's no real cost to not caching it.
    """
    skills: dict[str, Skill] = {}
    root = Path(settings.skills_root)
    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        skill_file = entry / "SKILL.md"
        if not entry.is_dir() or not skill_file.is_file():
            continue
        skill = _parse_skill_file(skill_file.read_text(encoding="utf-8"), f"{entry.name}/SKILL.md", entry.name)
        if skill.name in skills:
            raise ValueError(f"duplicate skill name '{skill.name}' (in {entry.name})")
        skills[skill.name] = skill
    return skills


def skills_index() -> str:
    """One 'name: description' line per skill, sorted by name for determinism."""
    return "\n".join(f"- {s.name}: {s.description}" for s in load_skills().values())


def get_skill(name: str) -> Skill | None:
    return load_skills().get(name)


def script_path(skill_name: str, script_name: str) -> Path:
    """Resolves a skill's script file on disk for get_script/run_script.
    Raises KeyError (not None) on either miss so callers can turn it
    directly into a tool-call error without a None-check layer.
    """
    skill = get_skill(skill_name)
    if skill is None:
        raise KeyError(f"No skill named '{skill_name}'. Available: {sorted(load_skills())}.")
    path = Path(settings.skills_root) / skill.dir_name / "scripts" / script_name
    if not path.is_file():
        raise KeyError(f"No script '{script_name}' under skill '{skill_name}'.")
    return path
