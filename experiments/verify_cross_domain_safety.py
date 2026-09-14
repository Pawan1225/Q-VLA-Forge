"""Independent verification for Sprint 5.14 cross-domain safety evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = ROOT / "results" / "safety" / "cross-domain"

PACKAGE = RESULTS_DIR / "sprint5-cross-domain-package.json"

CLAIMS = RESULTS_DIR / "sprint5-cross-domain-claim-matrix.json"

FIGURES = RESULTS_DIR / "sprint5-cross-domain-figure-index.json"

MECHANISM = RESULTS_DIR / "sprint5-cross-domain-lyapunov-mechanism.json"

CONSISTENCY = RESULTS_DIR / "sprint5-cross-domain-consistency.json"

REQUIRED_ARTIFACTS = (
    RESULTS_DIR / "sprint5-cross-domain-architecture.json",
    RESULTS_DIR / "sprint5-cross-domain-clean.json",
    RESULTS_DIR / "sprint5-cross-domain-gaussian.json",
    RESULTS_DIR / "sprint5-cross-domain-structured-state.json",
    RESULTS_DIR / "sprint5-cross-domain-action.json",
    MECHANISM,
    CONSISTENCY,
    CLAIMS,
    FIGURES,
    PACKAGE,
)


def _load(
    path: Path,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(f"expected JSON object: {path}")

    return payload


def _sha256(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def main() -> None:
    for path in REQUIRED_ARTIFACTS:
        if not path.exists():
            raise FileNotFoundError(path)

    package = _load(PACKAGE)

    claims = _load(CLAIMS)

    figures = _load(FIGURES)

    mechanism = _load(MECHANISM)

    consistency = _load(CONSISTENCY)

    # ---------------------------------------------------------
    # Package identity and execution controls
    # ---------------------------------------------------------

    if package["sprint"] != "5.14K":
        raise RuntimeError("unexpected package sprint")

    if not bool(package["analysis_only"]):
        raise RuntimeError("package must be analysis-only")

    if bool(package["new_training"]):
        raise RuntimeError("unexpected new training")

    if bool(package["new_principal_runs"]):
        raise RuntimeError("unexpected new principal runs")

    if bool(package["new_safety_episodes"]):
        raise RuntimeError("unexpected new safety episodes")

    # ---------------------------------------------------------
    # Source-manifest integrity
    # ---------------------------------------------------------

    source_manifest = package["source_manifest"]

    if not isinstance(
        source_manifest,
        list,
    ):
        raise TypeError("source_manifest must be list")

    if len(source_manifest) != 9:
        raise RuntimeError("expected 9 source artifacts")

    for record in source_manifest:
        if not isinstance(
            record,
            dict,
        ):
            raise TypeError("source manifest record must be dict")

        source_path = ROOT / str(record["path"])

        if not source_path.exists():
            raise FileNotFoundError(source_path)

        actual_hash = _sha256(source_path)

        expected_hash = str(record["sha256"])

        if actual_hash != expected_hash:
            raise RuntimeError("source hash mismatch: " f"{source_path}")

        if source_path.stat().st_size != int(record["size_bytes"]):
            raise RuntimeError("source size mismatch: " f"{source_path}")

    # ---------------------------------------------------------
    # Consistency evidence
    # ---------------------------------------------------------

    expected_consistency = {
        "clean_consistency": (
            2,
            2,
            0,
            True,
        ),
        "gaussian_consistency": (
            9,
            7,
            2,
            False,
        ),
        "structured_state_consistency": (
            3,
            3,
            0,
            True,
        ),
        "action_recovery_consistency": (
            3,
            3,
            0,
            True,
        ),
    }

    for key, expected in expected_consistency.items():
        record = consistency[key]

        if not isinstance(
            record,
            dict,
        ):
            raise TypeError(f"{key} must be dict")

        actual = (
            int(record["count"]),
            int(record["true"]),
            int(record["false"]),
            bool(record["all_consistent"]),
        )

        if actual != expected:
            raise RuntimeError(f"{key} mismatch: " f"{actual} != {expected}")

    activation = consistency["lyapunov_activation_consistency"]

    if not isinstance(
        activation,
        dict,
    ):
        raise TypeError("activation consistency must be dict")

    if not bool(activation["driving_activation_observed"]):
        raise RuntimeError("driving Lyapunov activation expected")

    if bool(activation["robotics_activation_observed"]):
        raise RuntimeError("robotics Lyapunov activation must remain false")

    if bool(activation["consistent"]):
        raise RuntimeError("Lyapunov activation must remain cross-domain inconsistent")

    # ---------------------------------------------------------
    # Mechanism invariants
    # ---------------------------------------------------------

    reconstruction = mechanism["global_reconstruction"]

    if not isinstance(
        reconstruction,
        dict,
    ):
        raise TypeError("mechanism reconstruction must be dict")

    reason_counts = reconstruction["intervention_reason_counts"]

    if not isinstance(
        reason_counts,
        dict,
    ):
        raise TypeError("reason_counts must be dict")

    expected_reasons = {
        "action_bound": 1493,
        "domain_constraint": 35729,
        "lyapunov_decrease": 1584,
        "none": 105194,
    }

    actual_reasons = {key: int(reason_counts[key]) for key in expected_reasons}

    if actual_reasons != expected_reasons:
        raise RuntimeError("mechanism reason-count mismatch")

    if int(reconstruction["strict_lyapunov_decrease_steps"]) != 70:
        raise RuntimeError("strict Lyapunov decrease mismatch")

    if int(reconstruction["selected_lower_than_perturbed_steps"]) != 4887:
        raise RuntimeError("selected-lower mismatch")

    if int(reconstruction["emergency_fallback_steps"]) != 0:
        raise RuntimeError("fallback mismatch")

    # ---------------------------------------------------------
    # Claim controls
    # ---------------------------------------------------------

    if int(claims["claim_count"]) != 12:
        raise RuntimeError("expected 12 claims")

    status_counts = claims["status_counts"]

    if not isinstance(
        status_counts,
        dict,
    ):
        raise TypeError("status_counts must be dict")

    expected_status_counts = {
        "supported": 5,
        "supported_with_limitation": 1,
        "not_supported": 6,
    }

    actual_status_counts = {
        key: int(status_counts[key]) for key in expected_status_counts
    }

    if actual_status_counts != expected_status_counts:
        raise RuntimeError("claim-status partition mismatch")

    blocked = {str(claim_id) for claim_id in claims["blocked_claim_ids"]}

    expected_blocked = {
        "S5-CD06",
        "S5-CD08",
        "S5-CD09",
        "S5-CD10",
        "S5-CD11",
        "S5-CD12",
    }

    if blocked != expected_blocked:
        raise RuntimeError("blocked claim set mismatch")

    # ---------------------------------------------------------
    # Figure provenance
    # ---------------------------------------------------------

    if int(figures["figure_count"]) != 6:
        raise RuntimeError("expected 6 figures")

    figure_records = figures["figures"]

    if not isinstance(
        figure_records,
        list,
    ):
        raise TypeError("figures must be list")

    if len(figure_records) != 6:
        raise RuntimeError("expected 6 figure records")

    seen_figure_ids: set[str] = set()

    for record in figure_records:
        if not isinstance(
            record,
            dict,
        ):
            raise TypeError("figure record must be dict")

        figure_id = str(record["figure_id"])

        if figure_id in seen_figure_ids:
            raise RuntimeError("duplicate figure ID")

        seen_figure_ids.add(figure_id)

        figure_path = ROOT / str(record["path"])

        if not figure_path.exists():
            raise FileNotFoundError(figure_path)

        if figure_path.stat().st_size <= 0:
            raise RuntimeError(f"empty figure: {figure_path}")

        source_artifacts = record["source_artifacts"]

        if not isinstance(
            source_artifacts,
            list,
        ):
            raise TypeError("figure sources must be list")

        for source in source_artifacts:
            source_path = ROOT / str(source)

            if not source_path.exists():
                raise FileNotFoundError(source_path)

    # ---------------------------------------------------------
    # Frozen section completeness
    # ---------------------------------------------------------

    expected_sections = {
        "5.14B",
        "5.14C",
        "5.14D",
        "5.14E",
        "5.14F",
        "5.14G",
        "5.14H",
        "5.14I",
        "5.14J",
    }

    frozen_sections = {str(section) for section in package["frozen_sections"]}

    if frozen_sections != expected_sections:
        raise RuntimeError("frozen-section set mismatch")

    print("=" * 80)

    print(" SPRINT 5.14L INDEPENDENT CROSS-DOMAIN VERIFICATION")

    print("=" * 80)

    print()

    print(
        "Required artifacts:",
        len(REQUIRED_ARTIFACTS),
    )

    print(
        "Source manifest entries:",
        len(source_manifest),
    )

    print(
        "Claims:",
        claims["claim_count"],
    )

    print(
        "Figures:",
        figures["figure_count"],
    )

    print()

    print("Artifact existence: PASS")

    print("Source hash integrity: PASS")

    print("Source size integrity: PASS")

    print("Clean consistency invariants: PASS")

    print("Gaussian consistency invariants: PASS")

    print("Structured-state consistency invariants: PASS")

    print("Action-recovery consistency invariants: PASS")

    print("Lyapunov activation asymmetry preserved: PASS")

    print("Mechanism totals: PASS")

    print("Claim partition: PASS")

    print("Blocked claim set: PASS")

    print("Figure provenance: PASS")

    print("Frozen-section completeness: PASS")

    print("No new training: PASS")

    print("No new principal execution: PASS")

    print()

    print("SPRINT 5.14L INDEPENDENT VERIFICATION: PASS")


if __name__ == "__main__":
    main()
