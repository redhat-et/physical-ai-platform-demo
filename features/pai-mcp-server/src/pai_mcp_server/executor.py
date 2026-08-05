from __future__ import annotations

import anyio

from pai_mcp_server import skills
from pai_mcp_server.config import settings
from pai_mcp_server.script_meta import ScriptMeta, parse_script_header


class ScriptError(Exception):
    """Argument validation or execution failure -- always safe to surface
    to the calling agent verbatim, since it never includes script source.
    """


def _coerce(value, param_type: str):
    if param_type == "integer":
        return int(value)
    if param_type == "number":
        return float(value)
    if param_type == "boolean":
        return bool(value)
    if param_type == "array":
        if not isinstance(value, list):
            raise TypeError(f"expected a list, got {type(value).__name__}")
        return [str(item) for item in value]
    return str(value)


def _validate_args(meta: ScriptMeta, args: dict) -> dict:
    declared = {p.name: p for p in meta.parameters}
    unknown = set(args) - set(declared)
    if unknown:
        raise ScriptError(f"unknown parameter(s): {sorted(unknown)}")

    validated: dict = {}
    for name, param in declared.items():
        if name in args:
            try:
                validated[name] = _coerce(args[name], param.type)
            except (TypeError, ValueError) as exc:
                raise ScriptError(f"parameter '{name}' must be of type {param.type}") from exc
        elif param.required:
            raise ScriptError(f"missing required parameter '{name}'")
        elif param.default is not None:
            validated[name] = param.default
    return validated


def _build_argv(script_file: str, validated: dict, meta: ScriptMeta) -> list[str]:
    """Named args become long flags (--name value; bare --name for a true
    boolean; --name item1 item2 ... for an array, matching argparse's
    nargs="*") rather than positional args, so scripts have a stable,
    self-documenting argv contract regardless of parameter order.
    """
    declared = {p.name: p for p in meta.parameters}
    argv = [script_file]
    for name, value in validated.items():
        param_type = declared[name].type
        if param_type == "boolean":
            if value:
                argv.append(f"--{name}")
        elif param_type == "array":
            if value:
                argv.append(f"--{name}")
                argv.extend(value)
        else:
            argv.extend([f"--{name}", str(value)])
    return argv


def get_script_meta(skill_name: str, script_name: str) -> ScriptMeta:
    path = skills.script_path(skill_name, script_name)
    return parse_script_header(path.read_text(encoding="utf-8"), f"{skill_name}/scripts/{script_name}")


async def run_script(skill_name: str, script_name: str, args: dict) -> dict:
    """Executes a skill script server-side after validating `args` against
    its declared header. Scripts must be executable with a shebang; invoked
    directly (no shell=True) so argument values can never be interpreted as
    shell syntax.

    Runs via anyio's native async subprocess support, not subprocess.run --
    this whole server is a single-threaded event loop shared by every
    connected agent (see server.py's call_tool and proxy.py's DownstreamProxy
    background task), so a blocking subprocess.run call here would freeze
    every other in-flight request for up to script_timeout_seconds.
    """
    path = skills.script_path(skill_name, script_name)
    meta = parse_script_header(path.read_text(encoding="utf-8"), f"{skill_name}/scripts/{script_name}")
    validated = _validate_args(meta, args or {})
    argv = _build_argv(str(path), validated, meta)
    try:
        with anyio.fail_after(settings.script_timeout_seconds):
            completed = await anyio.run_process(argv, check=False)
    except TimeoutError as exc:
        raise ScriptError(f"script timed out after {settings.script_timeout_seconds}s") from exc
    return {
        "exit_code": completed.returncode,
        "stdout": completed.stdout.decode(errors="replace"),
        "stderr": completed.stderr.decode(errors="replace"),
    }
