"""Audit synthetic dataset integrity before Sprint 4."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np

from q_vla_forge.data.contracts import TaskSample
from q_vla_forge.data.driving import SyntheticDrivingDataset
from q_vla_forge.data.robotics import SyntheticRoboticsDataset
from q_vla_forge.evaluation.pilot_readiness import (
    LOCKED_SEEDS,
    SplitAudit,
    overlap_count,
    sample_fingerprint,
    summarize_distribution,
)

OUTPUT_PATH = Path("results/pilot-readiness/dataset-audit.json")

TRAIN_SIZE = 512
VALIDATION_SIZE = 128
TEST_SIZE = 128


DatasetFactory = Callable[
    [int, int],
    Sequence[TaskSample],
]


def fingerprint_dataset(
    dataset: Sequence[TaskSample],
) -> set[str]:
    """Fingerprint every exact supervised sample in one split."""
    fingerprints: set[str] = set()

    for sample in dataset:
        fingerprints.add(
            sample_fingerprint(
                visual=sample.observation.visual,
                state=sample.observation.state,
                action=sample.target_action.values,
                language_goal=sample.observation.language_goal,
            )
        )

    return fingerprints


def collect_states(
    dataset: Sequence[TaskSample],
) -> np.ndarray:
    """Stack all state vectors from a dataset."""
    return np.stack(
        [sample.observation.state for sample in dataset],
        axis=0,
    )


def collect_actions(
    dataset: Sequence[TaskSample],
) -> np.ndarray:
    """Stack all target-action vectors from a dataset."""
    return np.stack(
        [sample.target_action.values for sample in dataset],
        axis=0,
    )


def arrays_are_finite(
    dataset: Sequence[TaskSample],
) -> bool:
    """Check visual, state, action, and safety arrays for finite values."""
    for sample in dataset:
        arrays = (
            sample.observation.visual,
            sample.observation.state,
            sample.target_action.values,
            sample.safety_constraints.lower_bounds,
            sample.safety_constraints.upper_bounds,
        )

        if not all(np.all(np.isfinite(array)) for array in arrays):
            return False

    return True


def actions_respect_constraints(
    dataset: Sequence[TaskSample],
) -> bool:
    """Check every target action against its declared safety bounds."""
    for sample in dataset:
        action = sample.target_action.values
        lower = sample.safety_constraints.lower_bounds
        upper = sample.safety_constraints.upper_bounds

        if np.any(action < lower):
            return False

        if np.any(action > upper):
            return False

    return True


def no_zero_variance_columns(
    values: np.ndarray,
) -> bool:
    """Require every feature dimension to vary across the split."""
    return bool(
        np.all(
            np.std(
                values,
                axis=0,
                ddof=0,
            )
            > 0.0
        )
    )


def audit_one_configuration(
    *,
    domain: str,
    seed: int,
    dataset_factory: DatasetFactory,
) -> dict:
    """Audit one domain/seed train-validation-test configuration."""
    train = dataset_factory(
        TRAIN_SIZE,
        seed,
    )

    validation = dataset_factory(
        VALIDATION_SIZE,
        seed + 1000,
    )

    test = dataset_factory(
        TEST_SIZE,
        seed + 2000,
    )

    train_fingerprints = fingerprint_dataset(train)

    validation_fingerprints = fingerprint_dataset(validation)

    test_fingerprints = fingerprint_dataset(test)

    split_audit = SplitAudit(
        domain=domain,
        seed=seed,
        train_size=len(train),
        validation_size=len(validation),
        test_size=len(test),
        train_unique=len(train_fingerprints),
        validation_unique=len(validation_fingerprints),
        test_unique=len(test_fingerprints),
        train_validation_overlap=overlap_count(
            train_fingerprints,
            validation_fingerprints,
        ),
        train_test_overlap=overlap_count(
            train_fingerprints,
            test_fingerprints,
        ),
        validation_test_overlap=overlap_count(
            validation_fingerprints,
            test_fingerprints,
        ),
    )

    train_states = collect_states(train)

    validation_states = collect_states(validation)

    test_states = collect_states(test)

    train_actions = collect_actions(train)

    validation_actions = collect_actions(validation)

    test_actions = collect_actions(test)

    finite_values_passed = all(
        arrays_are_finite(split)
        for split in (
            train,
            validation,
            test,
        )
    )

    action_constraints_passed = all(
        actions_respect_constraints(split)
        for split in (
            train,
            validation,
            test,
        )
    )

    state_variance_passed = all(
        no_zero_variance_columns(values)
        for values in (
            train_states,
            validation_states,
            test_states,
        )
    )

    action_variance_passed = all(
        no_zero_variance_columns(values)
        for values in (
            train_actions,
            validation_actions,
            test_actions,
        )
    )

    return {
        "domain": domain,
        "seed": seed,
        "split_audit": {
            **asdict(split_audit),
            "passed": split_audit.passed,
        },
        "finite_values_passed": finite_values_passed,
        "action_constraints_passed": action_constraints_passed,
        "state_variance_passed": state_variance_passed,
        "action_variance_passed": action_variance_passed,
        "distributions": {
            "train": {
                "state": asdict(summarize_distribution(train_states)),
                "action": asdict(summarize_distribution(train_actions)),
            },
            "validation": {
                "state": asdict(summarize_distribution(validation_states)),
                "action": asdict(summarize_distribution(validation_actions)),
            },
            "test": {
                "state": asdict(summarize_distribution(test_states)),
                "action": asdict(summarize_distribution(test_actions)),
            },
        },
    }


def main() -> None:
    results: list[dict] = []

    factories: tuple[
        tuple[
            str,
            DatasetFactory,
        ],
        ...,
    ] = (
        (
            "autonomous_driving",
            SyntheticDrivingDataset,
        ),
        (
            "robotics",
            SyntheticRoboticsDataset,
        ),
    )

    for (
        domain,
        dataset_factory,
    ) in factories:
        for seed in LOCKED_SEEDS:
            results.append(
                audit_one_configuration(
                    domain=domain,
                    seed=seed,
                    dataset_factory=dataset_factory,
                )
            )

    all_split_audits_passed = all(item["split_audit"]["passed"] for item in results)

    finite_values_passed = all(item["finite_values_passed"] for item in results)

    action_constraints_passed = all(
        item["action_constraints_passed"] for item in results
    )

    state_variance_passed = all(item["state_variance_passed"] for item in results)

    action_variance_passed = all(item["action_variance_passed"] for item in results)

    payload = {
        "audit": "Sprint 3.13 Dataset Integrity",
        "train_size": TRAIN_SIZE,
        "validation_size": VALIDATION_SIZE,
        "test_size": TEST_SIZE,
        "seeds": list(LOCKED_SEEDS),
        "split_seed_protocol": {
            "train": "seed",
            "validation": "seed + 1000",
            "test": "seed + 2000",
        },
        "results": results,
        "all_split_audits_passed": all_split_audits_passed,
        "finite_values_passed": finite_values_passed,
        "action_constraints_passed": action_constraints_passed,
        "state_variance_passed": state_variance_passed,
        "action_variance_passed": action_variance_passed,
        "overall_passed": all(
            (
                all_split_audits_passed,
                finite_values_passed,
                action_constraints_passed,
                state_variance_passed,
                action_variance_passed,
            )
        ),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("============================================")
    print(" SPRINT 3.13 DATASET AUDIT")
    print("============================================")

    print(f"Configurations: {len(results)}")

    print(
        "Split independence:",
        "PASS" if all_split_audits_passed else "FAIL",
    )

    print(
        "Finite values:",
        "PASS" if finite_values_passed else "FAIL",
    )

    print(
        "Action constraints:",
        "PASS" if action_constraints_passed else "FAIL",
    )

    print(
        "State variance:",
        "PASS" if state_variance_passed else "FAIL",
    )

    print(
        "Action variance:",
        "PASS" if action_variance_passed else "FAIL",
    )

    print(
        "Overall:",
        "PASS" if payload["overall_passed"] else "FAIL",
    )

    print(f"Artifact: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
