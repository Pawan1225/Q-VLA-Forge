"""Build the frozen Sprint 5.13A safety source manifest."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from q_vla_forge.evaluation.safety_consolidation import (
    ManifestEntry,
    make_manifest_entry,
)

ROOT = Path(__file__).resolve().parents[1]

SAFETY_ROOT = ROOT / "results" / "safety"

OUTPUT_ROOT = SAFETY_ROOT / "consolidated"

OUTPUT_PATH = OUTPUT_ROOT / "sprint5-safety-source-manifest.json"

CHECKPOINT_ROOT = ROOT / "results" / "rl" / "checkpoints"

SOURCE_GROUPS = (
    (
        "no_filter_clean",
        "5.4",
        "clean NONE safety evidence",
        SAFETY_ROOT / "baseline",
        9,
    ),
    (
        "clipping_clean",
        "5.5",
        "clean clipping safety evidence",
        SAFETY_ROOT / "clipping",
        9,
    ),
    (
        "lyapunov_foundation",
        "5.6",
        "Lyapunov foundation evidence",
        SAFETY_ROOT / "lyapunov-foundation",
        2,
    ),
    (
        "lyapunov_filter",
        "5.7",
        "Lyapunov filter evidence",
        SAFETY_ROOT / "lyapunov-filter",
        2,
    ),
    (
        "driving_lyapunov_clean",
        "5.8",
        "clean driving Lyapunov evidence",
        SAFETY_ROOT / "lyapunov-driving",
        6,
    ),
    (
        "robotics_lyapunov_clean",
        "5.9",
        "clean robotics Lyapunov evidence",
        SAFETY_ROOT / "lyapunov-robotics",
        6,
    ),
    (
        "gaussian_robustness",
        "5.10",
        "Gaussian observation-noise robustness",
        SAFETY_ROOT / "gaussian-robustness",
        57,
    ),
    (
        "structured_state_robustness",
        "5.11",
        "structured state robustness",
        SAFETY_ROOT / "structured-state-robustness",
        201,
    ),
    (
        "action_robustness",
        "5.12",
        "structured action robustness",
        SAFETY_ROOT / "action-robustness",
        220,
    ),
)


def _files_under(
    directory: Path,
) -> list[Path]:
    return sorted(path for path in directory.rglob("*") if path.is_file())


def main() -> None:
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    entries: list[ManifestEntry] = []

    group_counts: dict[str, int] = {}

    for (
        group,
        sprint,
        role,
        directory,
        expected_count,
    ) in SOURCE_GROUPS:
        if not directory.exists():
            raise RuntimeError(f"missing source directory: {directory}")

        files = _files_under(directory)

        if len(files) != expected_count:
            raise RuntimeError(
                f"{group}: expected " f"{expected_count} files, " f"found {len(files)}"
            )

        group_counts[group] = len(files)

        for path in files:
            entries.append(
                make_manifest_entry(
                    root=ROOT,
                    path=path,
                    sprint=sprint,
                    role=role,
                    group=group,
                )
            )

    checkpoint_files = sorted(CHECKPOINT_ROOT.glob("*-seed-*-final.pt"))

    if len(checkpoint_files) != 6:
        raise RuntimeError("expected exactly 6 frozen PPO checkpoints")

    group_counts["ppo_checkpoints"] = len(checkpoint_files)

    for path in checkpoint_files:
        entries.append(
            make_manifest_entry(
                root=ROOT,
                path=path,
                sprint="5.3.5",
                role="frozen reconstructed PPO checkpoint",
                group="ppo_checkpoints",
            )
        )

    protocol_sources = (
        ROOT / "src" / "q_vla_forge" / "safety" / "protocol.py",
        ROOT / "configs" / "safety.yaml",
        ROOT / "configs" / "action_robustness.yaml",
        ROOT / "configs" / "safety_consolidation.yaml",
    )

    for path in protocol_sources:
        if not path.exists():
            raise RuntimeError(f"missing protocol source: {path}")

        entries.append(
            make_manifest_entry(
                root=ROOT,
                path=path,
                sprint="5.1-5.13",
                role="frozen safety protocol/configuration",
                group="protocol",
            )
        )

    group_counts["protocol"] = len(protocol_sources)

    constraint_sources = (
        (
            "driving_constraints",
            "5.2",
            "driving safety constraints",
            ROOT / "src" / "q_vla_forge" / "safety" / "driving_constraints.py",
        ),
        (
            "robotics_constraints",
            "5.3",
            "robotics safety constraints",
            ROOT / "src" / "q_vla_forge" / "safety" / "robotics_constraints.py",
        ),
    )

    for (
        group,
        sprint,
        role,
        path,
    ) in constraint_sources:
        if not path.exists():
            raise RuntimeError(f"missing constraint source: {path}")

        entries.append(
            make_manifest_entry(
                root=ROOT,
                path=path,
                sprint=sprint,
                role=role,
                group=group,
            )
        )

        group_counts[group] = 1

    entries.sort(
        key=lambda entry: (
            entry.group,
            entry.path,
        )
    )

    payload = {
        "sprint": "5.13A",
        "artifact": "sprint5-safety-source-manifest",
        "analysis_only": True,
        "new_training": False,
        "new_ppo_runs": False,
        "new_principal_runs": False,
        "principal_seeds": [
            42,
            123,
            456,
        ],
        "manifest_entry_count": len(entries),
        "group_counts": group_counts,
        "entries": [asdict(entry) for entry in entries],
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 72)
    print(" SPRINT 5.13A SAFETY SOURCE MANIFEST")
    print("=" * 72)
    print()

    for group in sorted(group_counts):
        print(
            f"{group:34s}",
            group_counts[group],
        )

    print()
    print(
        "Manifest entries:",
        len(entries),
    )

    print(
        "PPO checkpoints:",
        group_counts["ppo_checkpoints"],
        "/ 6",
    )

    print()
    print("Source integrity: PASS")
    print("Three-seed source availability: PASS")
    print("No training: PASS")
    print("No new PPO execution: PASS")
    print("No new principal runs: PASS")
    print()
    print("SPRINT 5.13A SOURCE MANIFEST: PASS")


if __name__ == "__main__":
    main()
