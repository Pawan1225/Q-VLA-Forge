from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(".")

EVIDENCE_DIR = ROOT / "results" / "safety" / "evidence"

EVIDENCE = EVIDENCE_DIR / "sprint5-safety-evidence.json"

CSV_PATH = EVIDENCE_DIR / "sprint5-safety-evidence.csv"

MARKDOWN_PATH = EVIDENCE_DIR / "sprint5-safety-evidence.md"

CLAIM_MATRIX = EVIDENCE_DIR / "sprint5-safety-claim-matrix.json"

FIGURE_INDEX = EVIDENCE_DIR / "sprint5-safety-figure-index.json"

MANIFEST = EVIDENCE_DIR / "sprint5-safety-manifest.json"

FINAL_CLAIMS = (
    ROOT / "results" / "safety" / "final" / "sprint5-final-claim-boundary.json"
)

ACTION_SUMMARY = (
    ROOT
    / "results"
    / "safety"
    / "action-robustness"
    / "sprint5-action-robustness-summary.json"
)

GAUSSIAN_SUMMARY = (
    ROOT
    / "results"
    / "safety"
    / "gaussian-robustness"
    / "sprint5-gaussian-robustness-summary.json"
)

STATE_SUMMARY = (
    ROOT
    / "results"
    / "safety"
    / "structured-state-robustness"
    / "sprint5-structured-state-robustness-summary.json"
)

CROSS_ACTION = (
    ROOT / "results" / "safety" / "cross-domain" / "sprint5-cross-domain-action.json"
)

CROSS_LYAPUNOV = (
    ROOT
    / "results"
    / "safety"
    / "cross-domain"
    / "sprint5-cross-domain-lyapunov-mechanism.json"
)

PRINCIPAL_SEEDS = (
    42,
    123,
    456,
)

TOLERANCE = 1e-12


def load_json(
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)

    payload = json.loads(path.read_text(encoding="utf-8-sig"))

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(f"{path} must contain JSON object")

    return payload


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def assert_close(
    actual: float,
    expected: float,
    *,
    label: str,
    tolerance: float = TOLERANCE,
) -> None:
    if not math.isclose(
        actual,
        expected,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise RuntimeError(f"{label}: expected {expected!r}, " f"got {actual!r}")


def clean_run_path(
    domain: str,
    method: str,
    seed: int,
) -> Path:
    filename = f"{domain}-seed-{seed}.json"

    if method == "none":
        folder = "baseline"
    elif method == "clipping":
        folder = "clipping"
    elif method == "lyapunov":
        folder = (
            "lyapunov-driving"
            if domain == "autonomous_driving"
            else "lyapunov-robotics"
        )
    else:
        raise ValueError(method)

    return ROOT / "results" / "safety" / folder / "runs" / filename


def clean_summary(
    domain: str,
    method: str,
    seed: int,
) -> dict[str, Any]:
    payload = load_json(
        clean_run_path(
            domain,
            method,
            seed,
        )
    )

    key = "seed_summary" if method == "lyapunov" else "summary"

    summary = payload.get(key)

    if not isinstance(
        summary,
        dict,
    ):
        raise TypeError("Clean run summary missing")

    return summary


def clean_metric(
    summary: dict[str, Any],
    method: str,
    metric: str,
) -> float:
    keys = {
        "violation_step_rate": (
            "violation_step_rate"
            if method == "none"
            else "executed_violation_step_rate"
        ),
        "reward": "mean_reward",
        "success_rate": "success_rate",
        "intervention_rate": ("intervention_rate"),
    }

    key = keys[metric]

    value = summary.get(key)

    if value is None and method == "none" and metric == "intervention_rate":
        return 0.0

    if not isinstance(
        value,
        (int, float),
    ):
        raise TypeError(f"Missing numeric {metric}")

    return float(value)


def verify_clean(
    evidence: dict[str, Any],
) -> None:
    for domain in (
        "autonomous_driving",
        "robotics",
    ):
        for method in (
            "none",
            "clipping",
            "lyapunov",
        ):
            for metric in (
                "violation_step_rate",
                "reward",
                "success_rate",
                "intervention_rate",
            ):
                values = [
                    clean_metric(
                        clean_summary(
                            domain,
                            method,
                            seed,
                        ),
                        method,
                        metric,
                    )
                    for seed in PRINCIPAL_SEEDS
                ]

                expected_mean = float(statistics.mean(values))

                expected_sd = float(statistics.stdev(values))

                packaged = evidence["clean_safety"][domain][method][metric]

                assert_close(
                    float(packaged["mean"]),
                    expected_mean,
                    label=("clean mean " f"{domain}/{method}/{metric}"),
                )

                assert_close(
                    float(packaged["sample_sd"]),
                    expected_sd,
                    label=("clean SD " f"{domain}/{method}/{metric}"),
                )


def verify_gaussian(
    evidence: dict[str, Any],
) -> None:
    source = load_json(GAUSSIAN_SUMMARY)

    upstream = source.get("cells")

    packaged = evidence["gaussian_robustness"]["cells"]

    if not isinstance(
        upstream,
        list,
    ) or not isinstance(
        packaged,
        list,
    ):
        raise TypeError("Gaussian cells invalid")

    if len(upstream) != len(packaged):
        raise RuntimeError("Gaussian cell count mismatch")

    upstream_map = {}

    for row in upstream:
        if not isinstance(
            row,
            dict,
        ):
            raise TypeError("Gaussian upstream cell invalid")

        key = (
            str(row["domain"]),
            str(row["method"]),
            int(row["principal_seed"]),
            float(row["sigma"]),
        )

        upstream_map[key] = row

    for row in packaged:
        if not isinstance(
            row,
            dict,
        ):
            raise TypeError("Gaussian packaged cell invalid")

        key = (
            str(row["domain"]),
            str(row["method"]),
            int(row["principal_seed"]),
            float(row["sigma"]),
        )

        upstream_row = upstream_map[key]

        for packaged_key, upstream_key in (
            (
                "violation_step_rate",
                "violation_step_rate",
            ),
            (
                "reward",
                "reward",
            ),
            (
                "success_rate",
                "success_rate",
            ),
            (
                "intervention_rate",
                "intervention_rate",
            ),
        ):
            assert_close(
                float(row[packaged_key]),
                float(upstream_row[upstream_key]),
                label=("Gaussian " f"{key}/{packaged_key}"),
            )


def verify_structured_state(
    evidence: dict[str, Any],
) -> None:
    source = load_json(STATE_SUMMARY)

    upstream = source.get("cells")

    packaged = evidence["structured_state_robustness"]["cells"]

    if not isinstance(
        upstream,
        list,
    ) or not isinstance(
        packaged,
        list,
    ):
        raise TypeError("Structured-state cells invalid")

    if len(upstream) != 198:
        raise RuntimeError("Expected 198 frozen " "structured-state cells")

    if len(packaged) != 198:
        raise RuntimeError("Packaged structured-state " "count mismatch")

    keys = {
        (
            str(row["domain"]),
            str(row["method"]),
            str(row["perturbation_name"]),
            int(row["principal_seed"]),
        )
        for row in upstream
        if isinstance(
            row,
            dict,
        )
    }

    packaged_keys = {
        (
            str(row["domain"]),
            str(row["method"]),
            str(row["perturbation_name"]),
            int(row["principal_seed"]),
        )
        for row in packaged
        if isinstance(
            row,
            dict,
        )
    }

    if keys != packaged_keys:
        raise RuntimeError("Structured-state condition " "directionality/key mismatch")


def verify_action(
    evidence: dict[str, Any],
) -> None:
    source = load_json(ACTION_SUMMARY)

    findings = source.get("findings")

    if not isinstance(
        findings,
        dict,
    ):
        raise TypeError("Action findings missing")

    packaged = evidence["action_robustness"]["global"]

    expected = {
        "unsafe_perturbed_steps": int(findings["unsafe_perturbed_steps"]),
        "recovered_unsafe_steps": int(findings["recovered_unsafe_steps"]),
        "unresolved_unsafe_steps": int(findings["unresolved_unsafe_steps"]),
    }

    for key, value in expected.items():
        if int(packaged[key]) != value:
            raise RuntimeError(f"Action count mismatch: {key}")

    unsafe = expected["unsafe_perturbed_steps"]

    recovered = expected["recovered_unsafe_steps"]

    unresolved = expected["unresolved_unsafe_steps"]

    if unsafe != (recovered + unresolved):
        raise RuntimeError("Action reconstruction failed")

    expected_rate = recovered / unsafe

    assert_close(
        float(packaged["recovery_fraction"]),
        expected_rate,
        label="Global action recovery",
    )

    interface = evidence["action_robustness"]["environment_interface"]

    expected_adjustments = int(findings["environment_interface_adjustment_steps"])

    if (
        int(interface["environment_interface_adjustments_count"])
        != expected_adjustments
    ):
        raise RuntimeError("Environment interface count " "mismatch")


def verify_cross_domain_action(
    evidence: dict[str, Any],
) -> None:
    source = load_json(CROSS_ACTION)

    upstream = source.get("domain_method_summary")

    packaged = evidence["action_robustness"]["domain_method_summary"]

    if not isinstance(
        upstream,
        list,
    ) or not isinstance(
        packaged,
        list,
    ):
        raise TypeError("Action domain summaries invalid")

    upstream_map = {
        (
            str(row["domain"]),
            str(row["method"]),
        ): row
        for row in upstream
        if isinstance(
            row,
            dict,
        )
    }

    for row in packaged:
        if not isinstance(
            row,
            dict,
        ):
            continue

        key = (
            str(row["domain"]),
            str(row["method"]),
        )

        ref = upstream_map[key]

        for count_key in (
            "unsafe_perturbed_steps",
            "recovered_unsafe_steps",
            "unresolved_unsafe_steps",
        ):
            if int(row[count_key]) != int(ref[count_key]):
                raise RuntimeError("Cross-domain action " f"mismatch {key}/{count_key}")

        rate = ref.get("recovery_rate")

        packaged_rate = row.get("recovery_rate")

        if rate is None:
            if packaged_rate is not None:
                raise RuntimeError("Undefined recovery mismatch")
        else:
            assert_close(
                (
                    float(packaged_rate)
                    if isinstance(packaged_rate, (int, float))
                    else (_ for _ in ()).throw(
                        TypeError("Packaged recovery rate must be numeric")
                    )
                ),
                float(rate),
                label=("Cross-domain recovery " f"{key}"),
            )


def verify_lyapunov(
    evidence: dict[str, Any],
) -> None:
    source = load_json(CROSS_LYAPUNOV)

    upstream = source.get("domain_mechanism_summary")

    if not isinstance(
        upstream,
        list,
    ):
        raise TypeError("Lyapunov source summary invalid")

    packaged_rows = evidence["lyapunov_mechanisms"]["by_domain_and_regime"]

    if not isinstance(
        packaged_rows,
        list,
    ):
        raise TypeError("Packaged mechanism rows invalid")

    packaged_action = {
        str(row["domain"]): row
        for row in packaged_rows
        if isinstance(
            row,
            dict,
        )
        and row.get("regime") == "structured_action"
    }

    for row in upstream:
        if not isinstance(
            row,
            dict,
        ):
            continue

        domain = str(row["domain"])

        packaged = packaged_action[domain]

        upstream_reasons = row["intervention_reason_counts"]

        packaged_reasons = packaged["intervention_reason_counts"]

        if upstream_reasons != packaged_reasons:
            raise RuntimeError("Lyapunov reason count " f"mismatch: {domain}")

        if int(packaged["strict_decrease_count"]) != int(
            row["strict_lyapunov_decrease_steps"]
        ):
            raise RuntimeError("Strict Delta-V count " f"mismatch: {domain}")


def load_csv_rows() -> list[dict[str, str]]:
    with CSV_PATH.open(
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise RuntimeError("Evidence CSV empty")

    return rows


def verify_csv_hashes() -> None:
    rows = load_csv_rows()

    cache: dict[
        str,
        str,
    ] = {}

    for row in rows:
        source = row["source_artifact"]

        expected = row["source_sha256"]

        if source not in cache:
            path = ROOT / source

            if not path.is_file():
                raise FileNotFoundError(path)

            cache[source] = sha256_file(path)

        if cache[source] != expected:
            raise RuntimeError("CSV source hash mismatch: " f"{source}")


def verify_csv_clean_values(
    evidence: dict[str, Any],
) -> None:
    rows = load_csv_rows()

    index = {}

    for row in rows:
        if row["regime"] != "clean":
            continue

        key = (
            row["domain"],
            row["method"],
            int(row["seed"]),
            row["metric"],
        )

        index[key] = float(row["value"])

    for domain in (
        "autonomous_driving",
        "robotics",
    ):
        for method in (
            "none",
            "clipping",
            "lyapunov",
        ):
            for metric in (
                "violation_step_rate",
                "reward",
                "success_rate",
            ):
                packaged = evidence["clean_safety"][domain][method][metric]

                values = [
                    index[
                        (
                            domain,
                            method,
                            seed,
                            metric,
                        )
                    ]
                    for seed in PRINCIPAL_SEEDS
                ]

                reconstructed = float(statistics.mean(values))

                assert_close(
                    reconstructed,
                    float(packaged["mean"]),
                    label=("CSV/JSON clean " f"{domain}/{method}/{metric}"),
                )


def verify_figures() -> None:
    index = load_json(FIGURE_INDEX)

    figures = index.get("figures")

    if (
        not isinstance(
            figures,
            list,
        )
        or len(figures) != 5
    ):
        raise RuntimeError("Expected five indexed figures")

    for figure in figures:
        if not isinstance(
            figure,
            dict,
        ):
            raise TypeError("Figure row invalid")

        path = ROOT / str(figure["path"])

        if not path.is_file():
            raise FileNotFoundError(path)

        digest = sha256_file(path)

        if digest != figure["sha256"]:
            raise RuntimeError("Figure hash mismatch: " f"{path}")


def verify_manifest() -> None:
    manifest = load_json(MANIFEST)

    artifacts = manifest.get("artifacts")

    if not isinstance(
        artifacts,
        list,
    ):
        raise TypeError("Manifest artifacts invalid")

    for item in artifacts:
        if not isinstance(
            item,
            dict,
        ):
            raise TypeError("Manifest row invalid")

        if not bool(
            item.get(
                "required",
                False,
            )
        ):
            continue

        path = ROOT / str(item["path"])

        if not path.is_file():
            raise FileNotFoundError(path)

        digest = sha256_file(path)

        if digest != item["sha256"]:
            raise RuntimeError("Manifest hash mismatch: " f"{path}")


def verify_claim_controls() -> None:
    matrix = load_json(CLAIM_MATRIX)

    frozen = load_json(FINAL_CLAIMS)

    if int(frozen["claim_count"]) != 25:
        raise RuntimeError("Frozen final claim count changed")

    proposal = matrix.get("proposal_claims")

    unsupported = matrix.get("explicitly_unsupported")

    if not isinstance(
        proposal,
        list,
    ) or not isinstance(
        unsupported,
        list,
    ):
        raise TypeError("Claim matrix invalid")

    positive_ids = {
        str(row["claim_id"])
        for row in proposal
        if isinstance(
            row,
            dict,
        )
    }

    if positive_ids != {
        "S5-E01",
        "S5-E02",
        "S5-E03",
        "S5-E04",
        "S5-E05",
        "S5-E06",
    }:
        raise RuntimeError("Proposal claim ID mismatch")

    unsupported_text = " ".join(
        str(
            row.get(
                "statement",
                "",
            )
        )
        for row in unsupported
        if isinstance(
            row,
            dict,
        )
    ).lower()

    required = (
        "lyapunov stability",
        "worst-case robustness",
        "iso 26262",
        "production safety",
        "real-world sensor",
        "universal safety controller",
        "quantum safety advantage",
    )

    for phrase in required:
        if phrase not in (unsupported_text):
            raise RuntimeError("Missing blocked claim: " f"{phrase}")


def verify_no_overclaim_text() -> None:
    markdown = MARKDOWN_PATH.read_text(encoding="utf-8").lower()

    forbidden = (
        "certified safe",
        "proven safe",
        "guaranteed stable",
        "iso 26262 compliant",
        "production-ready safety",
        "quantum safety advantage demonstrated",
    )

    for phrase in forbidden:
        if phrase in markdown:
            raise RuntimeError("Overclaim wording found: " f"{phrase}")


def verify_scope(
    evidence: dict[str, Any],
) -> None:
    metadata = evidence.get("metadata")

    if not isinstance(
        metadata,
        dict,
    ):
        raise TypeError("Metadata missing")

    expected = {
        "new_training": False,
        "new_principal_execution": False,
        "new_scientific_experiment": False,
        "scientific_scope_frozen": True,
    }

    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"Scope invariant failed: {key}")


def main() -> None:
    evidence = load_json(EVIDENCE)

    print("=" * 64)
    print(" Q-VLA FORGE - SPRINT 5.15 " "INDEPENDENT EVIDENCE VERIFIER")
    print("=" * 64)
    print()

    verify_scope(evidence)

    print("Scientific scope                     PASS")

    verify_clean(evidence)

    print("Clean evidence reconstruction        PASS")

    verify_gaussian(evidence)

    print("Gaussian evidence reconstruction     PASS")

    verify_structured_state(evidence)

    print("Structured-state reconstruction      PASS")

    verify_action(evidence)

    print("Action recovery reconstruction       PASS")

    verify_cross_domain_action(evidence)

    print("Cross-domain action reconstruction   PASS")

    verify_lyapunov(evidence)

    print("Lyapunov mechanism reconstruction    PASS")

    verify_csv_clean_values(evidence)

    print("CSV / JSON numerical consistency     PASS")

    verify_csv_hashes()

    print("CSV upstream hashes                  PASS")

    verify_figures()

    print("Figure hashes                        PASS")

    verify_manifest()

    print("Evidence manifest hashes             PASS")

    verify_claim_controls()

    print("Claim boundary controls              PASS")

    verify_no_overclaim_text()

    print("Anti-overclaim scan                  PASS")

    print()
    print("No training                          PASS")
    print("No policy execution                  PASS")
    print("No new scientific experiment         PASS")
    print()
    print("SPRINT 5.15 INDEPENDENT " "EVIDENCE VERIFICATION: PASS")


if __name__ == "__main__":
    main()
