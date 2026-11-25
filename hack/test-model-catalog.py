# /// script
# requires-python = ">=3.11"
# dependencies = ["ruamel.yaml==0.18.16"]
# ///

import io
import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
import unittest

from ruamel.yaml import YAML


SCRIPT = Path(__file__).with_name("update-model-catalog.py")
SHA = "a" * 40
REPO = "example/robot"
HUB = "https://huggingface.co"


class CatalogRefreshTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        self.allowlist = directory / "allowlist.yaml"
        self.catalog = directory / "catalog.yaml"
        self.allowlist.write_text(
            "models:\n- name: Robot\n  repo: example/robot\n"
            "  overrides:\n    description: Local summary\n"
            "    customProperties:\n      platform_note: local metadata\n"
            "  supplement: 'Requires matching normalization statistics.'\n",
            encoding="utf-8",
        )
        self.original = b"source: Physical AI\nmodels:\n- name: MockLM\n  readme: Local card\n  artifacts: []\n"
        self.catalog.write_bytes(self.original)
        self.responses = {
            f"{HUB}/api/models/{REPO}/revision/main": self.info(SHA),
            f"{HUB}/{REPO}/raw/{SHA}/README.md": (
                "---\nlicense: apache-2.0\nlanguage: [en]\n---\n"
                "# Robot\n\n![Example](docs/result.png)\nHard break here.  \nNext line.\n"
            ),
        }

    def info(self, revision):
        return json.dumps({"sha": revision, "author": "example", "pipeline_tag": "robotics"})

    def run_update(self, *flags):
        def open_fixture(url, *, timeout):
            if url not in self.responses:
                raise OSError(f"No fixture for {url}")
            return io.BytesIO(self.responses[url].encode())

        main = runpy.run_path(str(SCRIPT))["main"]
        return main(
            ["--allowlist", str(self.allowlist), "--catalog", str(self.catalog), *flags],
            opener=open_fixture,
        )

    def test_refresh_preserves_manual_cards_and_pins_generated_cards(self):
        self.assertEqual(self.run_update(), 0)
        catalog = YAML(typ="safe").load(self.catalog)
        mock, model = catalog["models"]
        self.assertEqual(mock, {"name": "MockLM", "readme": "Local card", "artifacts": []})
        self.assertEqual(model["name"], "Robot")
        self.assertEqual(model["description"], "Local summary")
        self.assertEqual(model["license"], "apache-2.0")
        self.assertEqual(model["language"], ["en"])
        self.assertEqual(model["tasks"], ["robotics"])
        self.assertEqual(model["artifacts"], [{"uri": "hf://example/robot:" + SHA}])
        self.assertEqual(
            model["customProperties"]["platform_note"],
            {"metadataType": "MetadataStringValue", "string_value": "local metadata"},
        )
        self.assertIn(f"![Example]({HUB}/{REPO}/resolve/{SHA}/docs/result.png)", model["readme"])
        self.assertIn("## Physical AI notes\n\nRequires matching normalization statistics.", model["readme"])
        self.assertNotIn("license: apache", model["readme"])
        self.assertIn("Hard break here.\\\nNext line.", model["readme"])
        first = self.catalog.read_bytes()
        self.assertEqual(self.run_update(), 0)
        self.assertEqual(self.catalog.read_bytes(), first)

    def test_refresh_checks_and_updates_an_advanced_revision(self):
        self.assertEqual(self.run_update(), 0)
        old_catalog = self.catalog.read_bytes()
        revision = "b" * 40
        self.responses[f"{HUB}/api/models/{REPO}/revision/main"] = self.info(revision)
        self.responses[f"{HUB}/{REPO}/raw/{revision}/README.md"] = "# New card\n"
        self.assertEqual(self.run_update("--check"), 1)
        self.assertEqual(self.catalog.read_bytes(), old_catalog)
        self.assertEqual(self.run_update(), 0)
        model = YAML(typ="safe").load(self.catalog)["models"][1]
        self.assertEqual(model["artifacts"], [{"uri": f"hf://{REPO}:{revision}"}])
        self.assertIn("# New card", model["readme"])
        self.assertEqual(self.run_update("--check"), 0)

    def test_failure_on_second_model_leaves_catalog_untouched(self):
        with self.allowlist.open("a", encoding="utf-8") as config:
            config.write("- name: Second robot\n  repo: example/another-robot\n")
        self.assertEqual(self.run_update(), 2)
        self.assertEqual(self.catalog.read_bytes(), self.original)

    def test_removing_allowlist_entry_preserves_manual_card(self):
        self.assertEqual(self.run_update(), 0)
        self.allowlist.write_text("models: []\n", encoding="utf-8")
        self.assertEqual(self.run_update(), 0)
        self.assertEqual(self.catalog.read_bytes(), self.original)

    def test_allowlist_cannot_take_over_an_unmanaged_card(self):
        self.allowlist.write_text("models:\n- name: MockLM\n  repo: example/robot\n", encoding="utf-8")
        self.assertEqual(self.run_update(), 2)
        self.assertEqual(self.catalog.read_bytes(), self.original)


if __name__ == "__main__":
    unittest.main()
