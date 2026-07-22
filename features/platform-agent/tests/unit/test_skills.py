import pytest

from platform_agent.tools.skills import (
    Skill,
    _load_skills,
    _parse_skill_file,
    get_skill,
    skills_index,
)


def test_parse_skill_file_valid():
    text = "---\nname: foo\ndescription: does foo things\n---\nBody line one.\nBody line two.\n"
    skill = _parse_skill_file(text, "foo.md")
    assert skill == Skill(
        name="foo",
        description="does foo things",
        body="Body line one.\nBody line two.",
    )


def test_parse_skill_file_missing_frontmatter():
    with pytest.raises(ValueError, match="missing leading"):
        _parse_skill_file("Just a body, no frontmatter.\n", "bad.md")


@pytest.mark.parametrize("missing_field", ["name", "description"])
def test_parse_skill_file_missing_required_field(missing_field):
    fields = {"name": "foo", "description": "does foo things"}
    del fields[missing_field]
    frontmatter = "\n".join(f"{k}: {v}" for k, v in fields.items())
    text = f"---\n{frontmatter}\n---\nBody.\n"
    with pytest.raises(ValueError, match=missing_field):
        _parse_skill_file(text, "bad.md")


def test_load_skills_returns_expected_names():
    skills = _load_skills()
    assert set(skills.keys()) == {"deploy-model", "datasets", "fine-tuning", "new-model-runtime"}


def test_skills_index_lists_all_skills():
    index = skills_index()
    for name in ("deploy-model", "datasets", "fine-tuning", "new-model-runtime"):
        assert name in index


def test_get_skill_returns_body():
    result = get_skill.invoke({"name": "datasets"})
    assert "Franka Panda" in result


def test_get_skill_unknown_name_lists_available():
    result = get_skill.invoke({"name": "nonexistent"})
    assert "No skill named 'nonexistent'" in result
    assert "datasets" in result
    assert "deploy-model" in result
    assert "fine-tuning" in result
    assert "new-model-runtime" in result
