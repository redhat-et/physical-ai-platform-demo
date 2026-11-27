"""Stage a local serving copy of a mounted LIBERO checkpoint's metadata."""

import json
import sys
from pathlib import Path


def stage_checkpoint(snapshot: Path, target: Path) -> None:
    config = json.loads((snapshot / "config.json").read_text())
    # v0.34.0 validates config.json before merging deploy settings. It returns
    # the whole generated chunk, so its chunk size must match n_action_steps.
    config["chunk_size"] = config["n_action_steps"]
    # Translate LeRobot's generated empty camera to a padded, masked slot in
    # the deploy config (three slots, two named observation keys).
    config["empty_cameras"] = 0
    (target / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for artifact in snapshot.iterdir():
        if artifact.name != "config.json":
            (target / artifact.name).symlink_to(artifact.resolve(strict=True))


if __name__ == "__main__":
    stage_checkpoint(Path(sys.argv[1]), Path(sys.argv[2]))
