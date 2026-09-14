"""Audit available Sprint 5 safety evidence before manifest freeze."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SAFETY_ROOT = ROOT / "results" / "safety"

EXPECTED_DIRS = (
    "baseline",
    "clipping",
    "lyapunov-foundation",
    "lyapunov-filter",
    "lyapunov-driving",
    "lyapunov-robotics",
    "gaussian-robustness",
    "structured-state-robustness",
    "action-robustness",
)


def main() -> None:
    print("=" * 72)
    print(" SPRINT 5.13A SAFETY SOURCE INVENTORY")
    print("=" * 72)
    print()

    for name in EXPECTED_DIRS:
        path = SAFETY_ROOT / name

        files = (
            sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
            if path.exists()
            else []
        )

        status = "PRESENT" if files else "MISSING"

        print(f"{name:34s}" f"{status:10s}" f"{len(files):6d} files")

    print()
    print("PPO checkpoints:")

    checkpoint_root = ROOT / "results" / "rl" / "checkpoints"

    checkpoint_files = sorted(checkpoint_root.glob("*-seed-*-final.pt"))

    for path in checkpoint_files:
        print(
            " ",
            path.relative_to(ROOT),
        )

    print()
    print(
        "Checkpoint count:",
        len(checkpoint_files),
    )

    print()
    print("Safety package source files:")

    safety_package = ROOT / "src" / "q_vla_forge" / "safety"

    for path in sorted(safety_package.glob("*.py")):
        print(
            " ",
            path.name,
        )

    print()
    print("SPRINT 5.13A SOURCE INVENTORY: COMPLETE")


if __name__ == "__main__":
    main()
