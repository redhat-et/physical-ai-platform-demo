from __future__ import annotations

import re
from dataclasses import dataclass, field

import yaml

# Matches a '# ---' … '# ---' delimited block anywhere in the file (so it
# can follow an optional shebang line), mirroring SKILL.md's '---' YAML
# frontmatter convention but in comment form so it works inside any
# '#'-comment scripting language (bash, python, ...).
_HEADER_RE = re.compile(r"^# ---\n((?:#.*\n)*?)# ---\n", re.MULTILINE)
_COMMENT_PREFIX_RE = re.compile(r"^#\s?", re.MULTILINE)


@dataclass(frozen=True)
class ScriptParam:
    name: str
    type: str
    required: bool = False
    default: object | None = None
    description: str = ""


@dataclass(frozen=True)
class ScriptMeta:
    description: str
    parameters: list[ScriptParam] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "description": self.description,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "required": p.required,
                    "default": p.default,
                    "description": p.description,
                }
                for p in self.parameters
            ],
        }


def parse_script_header(text: str, filename: str) -> ScriptMeta:
    """Parses the '# ---' metadata header every skill script must declare --
    this is the entire contract get_script hands back to an agent, and the
    entire contract run_script validates arguments against, without either
    tool ever returning the script body itself.
    """
    match = _HEADER_RE.search(text)
    if not match:
        raise ValueError(f"{filename}: missing '# ---' metadata header block")
    raw_yaml = _COMMENT_PREFIX_RE.sub("", match.group(1))
    meta = yaml.safe_load(raw_yaml) or {}
    if not meta.get("description"):
        raise ValueError(f"{filename}: header missing required 'description' field")

    params: list[ScriptParam] = []
    for raw_param in meta.get("parameters", []) or []:
        for required_field in ("name", "type"):
            if not raw_param.get(required_field):
                raise ValueError(f"{filename}: parameter missing required '{required_field}' field")
        params.append(ScriptParam(
            name=raw_param["name"],
            type=raw_param["type"],
            required=bool(raw_param.get("required", False)),
            default=raw_param.get("default"),
            description=raw_param.get("description", ""),
        ))
    return ScriptMeta(description=meta["description"], parameters=params)
