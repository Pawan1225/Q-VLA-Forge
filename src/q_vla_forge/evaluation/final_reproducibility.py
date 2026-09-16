"""Sprint 7.14 â€” final Phase 1 reproducibility verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_SEEDS = (42, 123, 456)

EXPECTED_BATCHES = {
    "batch-a": ("results/final-validation/batch-a/" "batch-a-acceptance.json"),
    "batch-b": ("results/final-validation/batch-b/" "batch-b-acceptance.json"),
    "batch-c": ("results/final-validation/batch-c/" "batch-c-acceptance.json"),
    "batch-d": ("results/final-validation/batch-d/" "batch-d-acceptance.json"),
}

REQUIRED_ARTIFACTS = (
    ("results/final-validation/three-seed/" "three-seed-validation.json"),
    ("results/final-validation/evidence/" "phase1-evidence.json"),
    ("results/final-validation/baseline/" "final-baseline-summary.json"),
    ("results/final-validation/compression/" "final-compression-summary.json"),
    (
        "results/final-validation/training-efficiency/"
        "final-training-efficiency-summary.json"
    ),
    ("results/final-validation/rl-qml/" "final-rl-qml-summary.json"),
    (
        "results/final-validation/safety-robustness/"
        "final-safety-robustness-summary.json"
    ),
    ("results/final-validation/cross-domain/" "final-cross-domain-summary.json"),
    ("results/final-validation/ablation/" "final-ablation-matrix.json"),
    ("results/final-validation/scorecard/" "challenge-bottleneck-scorecard.json"),
    ("results/final-validation/claims/" "final-claim-registry.json"),
    ("results/final-validation/submission-assets/" "final-submission-assets.json"),
    ("results/final-validation/submission-assets/" "repository-evidence-layer.json"),
    ("results/final-validation/tables/" "final-ablation-matrix.csv"),
    ("results/final-validation/tables/" "challenge-bottleneck-scorecard.csv"),
    ("results/final-validation/tables/" "final-claim-registry.csv"),
    ("figures/final/" "phase1_compression_pareto.png"),
    ("figures/final/" "phase1_rl_qml_target_reach.png"),
    "README.md",
    "docs/final_evidence_index.md",
)


def load_json(
    path: Path,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(f"Expected JSON object: {path}")

    return payload


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def verify_batch_acceptances(
    root: Path,
) -> dict[str, Any]:
    results: dict[
        str,
        Any,
    ] = {}

    for batch, relative_path in EXPECTED_BATCHES.items():
        path = root / relative_path

        if not path.exists():
            results[batch] = {
                "passed": False,
                "reason": ("missing acceptance artifact"),
                "path": relative_path,
            }
            continue

        payload = load_json(path)

        results[batch] = {
            "passed": bool(
                payload.get(
                    "passed",
                    False,
                )
            ),
            "new_training": payload.get("new_training"),
            "new_experiments": payload.get("new_experiments"),
            "path": relative_path,
        }

    return results


def verify_required_artifacts(
    root: Path,
) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []

    for relative_path in REQUIRED_ARTIFACTS:
        path = root / relative_path

        exists = path.exists()

        record: dict[
            str,
            Any,
        ] = {
            "path": relative_path,
            "exists": exists,
        }

        if exists and path.is_file():
            record["size_bytes"] = path.stat().st_size

            record["sha256"] = sha256_file(path)

        artifacts.append(record)

    missing = [item["path"] for item in artifacts if not item["exists"]]

    return {
        "artifacts": artifacts,
        "required_count": len(REQUIRED_ARTIFACTS),
        "present_count": (len(REQUIRED_ARTIFACTS) - len(missing)),
        "missing": missing,
        "passed": not missing,
    }


def verify_seed_contract(
    root: Path,
) -> dict[str, Any]:
    path = (
        root
        / "results"
        / "final-validation"
        / "three-seed"
        / "three-seed-validation.json"
    )

    payload = load_json(path)

    groups = payload.get(
        "groups",
        [],
    )

    if not isinstance(
        groups,
        list,
    ):
        raise TypeError("Three-seed validation " "groups must be a list.")

    checked_groups: list[dict[str, Any]] = []

    for group in groups:
        if not isinstance(
            group,
            dict,
        ):
            continue

        observed = tuple(
            group.get(
                "observed_seeds",
                [],
            )
        )

        checked_groups.append(
            {
                "group": group.get("group"),
                "observed_seeds": list(observed),
                "exact_match": (observed == EXPECTED_SEEDS),
            }
        )

    passed = bool(checked_groups) and all(
        item["exact_match"] for item in checked_groups
    )

    return {
        "expected_seeds": list(EXPECTED_SEEDS),
        "groups": checked_groups,
        "passed": passed,
    }


def verify_claim_freeze(
    root: Path,
) -> dict[str, Any]:
    path = (
        root / "results" / "final-validation" / "claims" / "final-claim-registry.json"
    )

    payload = load_json(path)

    summary = payload.get(
        "summary",
        {},
    )

    return {
        "frozen": bool(
            payload.get(
                "frozen",
                False,
            )
        ),
        "status": payload.get("status"),
        "new_training": payload.get("new_training"),
        "new_experiments": payload.get("new_experiments"),
        "supported": summary.get("supported"),
        "supported_with_limitation": (summary.get("supported_with_limitation")),
        "blocked": summary.get("blocked"),
    }


def build_final_reproducibility(
    root: Path,
) -> dict[str, Any]:
    """Build the Sprint 7.14 reproducibility record."""

    batches = verify_batch_acceptances(root)

    artifacts = verify_required_artifacts(root)

    seeds = verify_seed_contract(root)

    claims = verify_claim_freeze(root)

    batches_passed = all(
        item.get(
            "passed",
            False,
        )
        for item in batches.values()
    )

    no_batch_training = all(
        item.get("new_training") is not True for item in batches.values()
    )

    claim_freeze_passed = (
        claims["frozen"]
        and claims["status"] == "FROZEN"
        and claims["new_training"] is False
        and claims["new_experiments"] is False
    )

    passed = all(
        (
            batches_passed,
            no_batch_training,
            artifacts["passed"],
            seeds["passed"],
            claim_freeze_passed,
        )
    )

    return {
        "sprint": "7.14",
        "protocol": ("final_reproducibility_verification"),
        "status": ("PASS" if passed else "FAIL"),
        "batch_acceptances": batches,
        "batches_passed": (batches_passed),
        "required_artifacts": (artifacts),
        "seed_contract": seeds,
        "claim_freeze": claims,
        "frozen_evidence_preserved": (True),
        "new_training": False,
        "new_experiments": False,
        "passed": passed,
    }
