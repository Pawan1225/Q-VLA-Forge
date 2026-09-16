"""Sprint 7.1 — final three-seed Phase 1 evidence validation."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REQUIRED_SEEDS: tuple[int, ...] = (42, 123, 456)

REQUIRED_GROUPS: tuple[str, ...] = (
    "baseline",
    "compression",
    "training_efficiency",
    "ppo",
    "qml",
    "safety",
    "robustness",
)

GROUP_PATH_PREFIXES: dict[str, tuple[str, ...]] = {
    "baseline": (
        "results/driving-baseline-",
        "results/robotics-baseline-",
        "results/baseline-validation-summary.json",
    ),
    "compression": ("results/compression/",),
    "training_efficiency": ("results/training/",),
    "ppo": ("results/rl/ppo/",),
    "qml": (
        "results/rl/qml/",
        "results/rl/ablation/",
    ),
    "safety": (
        "results/safety/baseline/",
        "results/safety/clipping/",
        "results/safety/lyapunov-driving/",
        "results/safety/lyapunov-robotics/",
    ),
    "robustness": (
        "results/safety/gaussian-robustness/",
        "results/safety/structured-state-robustness/",
        "results/safety/action-robustness/",
    ),
}

RESULT_HINTS: tuple[str, ...] = (
    "mse",
    "mae",
    "loss",
    "reward",
    "success",
    "latency",
    "compression",
    "efficiency",
    "violation",
    "intervention",
    "runtime",
    "steps",
    "episodes",
    "target",
)


@dataclass(frozen=True)
class SeedEvidence:
    """Observed evidence for one required experiment group."""

    group: str
    required_seeds: tuple[int, ...]
    observed_seeds: tuple[int, ...]
    evidence_paths: tuple[str, ...]
    seed_sources: dict[int, tuple[str, ...]]

    @property
    def missing_seeds(self) -> tuple[int, ...]:
        return tuple(
            seed for seed in self.required_seeds if seed not in self.observed_seeds
        )

    @property
    def extra_seeds(self) -> tuple[int, ...]:
        return tuple(
            seed for seed in self.observed_seeds if seed not in self.required_seeds
        )

    @property
    def complete(self) -> bool:
        return not self.missing_seeds

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)

        payload["missing_seeds"] = list(self.missing_seeds)

        payload["extra_seeds"] = list(self.extra_seeds)

        payload["complete"] = self.complete

        payload["seed_sources"] = {
            str(seed): list(paths) for seed, paths in self.seed_sources.items()
        }

        return payload


@dataclass(frozen=True)
class ThreeSeedValidation:
    """Final Sprint 7.1 validation result."""

    groups: tuple[SeedEvidence, ...]

    @property
    def complete_groups(self) -> int:
        return sum(group.complete for group in self.groups)

    @property
    def incomplete_groups(self) -> int:
        return len(self.groups) - self.complete_groups

    @property
    def passed(self) -> bool:
        names = tuple(group.group for group in self.groups)

        return names == REQUIRED_GROUPS and self.incomplete_groups == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sprint": "7.1",
            "protocol": "final_three_seed_validation",
            "scope": "frozen_phase1_evidence_only",
            "required_seeds": list(REQUIRED_SEEDS),
            "required_groups": list(REQUIRED_GROUPS),
            "complete_groups": (self.complete_groups),
            "incomplete_groups": (self.incomplete_groups),
            "passed": self.passed,
            "groups": [group.to_dict() for group in self.groups],
        }


def is_generated_validation_path(
    root: Path,
    path: Path,
) -> bool:
    """Prevent Sprint 7 outputs from validating themselves."""

    relative = path.relative_to(root).as_posix().lower()

    return relative.startswith("results/final-validation/")


def path_seed(
    path: Path,
) -> set[int]:
    """Extract an explicit locked seed encoded in a file path."""

    matches = re.findall(
        r"seed[-_](42|123|456)(?:\D|$)",
        path.as_posix().lower(),
    )

    return {int(value) for value in matches}


def has_result_content(
    value: Any,
) -> bool:
    """Return True when a payload contains result-like fields."""

    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower()

            if any(hint in normalized for hint in RESULT_HINTS):
                return True

            if has_result_content(item):
                return True

    elif isinstance(value, list):
        return any(has_result_content(item) for item in value)

    return False


def extract_observed_seeds(
    value: Any,
    *,
    allow_aggregate_seeds: bool,
) -> set[int]:
    """Extract executed seeds without treating requirements as evidence."""

    discovered: set[int] = set()

    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower()

            if normalized == "seed" and isinstance(item, int):
                discovered.add(item)

            elif (normalized == "evaluated_seeds" and isinstance(item, list)) or (
                normalized == "seeds"
                and allow_aggregate_seeds
                and isinstance(item, list)
            ):
                discovered.update(seed for seed in item if isinstance(seed, int))

            discovered.update(
                extract_observed_seeds(
                    item,
                    allow_aggregate_seeds=(allow_aggregate_seeds),
                )
            )

    elif isinstance(value, list):
        for item in value:
            discovered.update(
                extract_observed_seeds(
                    item,
                    allow_aggregate_seeds=(allow_aggregate_seeds),
                )
            )

    return discovered


def matches_group(
    path: Path,
    group: str,
) -> bool:
    """Match evidence only against canonical frozen path scopes."""

    relative = path.as_posix().lower()

    return any(relative.startswith(prefix) for prefix in GROUP_PATH_PREFIXES[group])


def inspect_group(
    root: Path,
    group: str,
) -> SeedEvidence:
    """Inspect frozen JSON evidence for one canonical group."""

    results = root / "results"

    if not results.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results}")

    sources: dict[int, set[str]] = {}
    evidence_paths: set[str] = set()

    for path in sorted(results.rglob("*.json")):
        if is_generated_validation_path(
            root,
            path,
        ):
            continue

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (
            json.JSONDecodeError,
            OSError,
        ):
            continue

        relative = path.relative_to(root)

        if not matches_group(
            relative,
            group,
        ):
            continue

        discovered = path_seed(relative)

        is_manifest = "manifest" in relative.as_posix().lower()

        allow_aggregate = not is_manifest and has_result_content(payload)

        discovered.update(
            extract_observed_seeds(
                payload,
                allow_aggregate_seeds=(allow_aggregate),
            )
        )

        if not discovered:
            continue

        relative_text = relative.as_posix()

        evidence_paths.add(relative_text)

        for seed in discovered:
            sources.setdefault(
                seed,
                set(),
            ).add(relative_text)

    observed = tuple(sorted(sources))

    seed_sources = {
        seed: tuple(sorted(paths)) for seed, paths in sorted(sources.items())
    }

    return SeedEvidence(
        group=group,
        required_seeds=(REQUIRED_SEEDS),
        observed_seeds=observed,
        evidence_paths=tuple(sorted(evidence_paths)),
        seed_sources=(seed_sources),
    )


def validate_three_seed_evidence(
    root: Path,
) -> ThreeSeedValidation:
    """Validate locked three-seed coverage across Phase 1."""

    groups = tuple(
        inspect_group(
            root,
            group,
        )
        for group in REQUIRED_GROUPS
    )

    return ThreeSeedValidation(groups=groups)
