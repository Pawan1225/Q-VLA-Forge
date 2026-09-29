"""Independent verification for Sprint 7.5 safety ablation."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "results" / "final-validation" / "safety-ablation"

BASELINE_SOURCE = (
    ROOT / "results" / "safety" / "baseline" / "sprint5-no-filter-safety-summary.json"
)

CLIPPING_SOURCE = (
    ROOT / "results" / "safety" / "clipping" / "sprint5-clipping-safety-summary.json"
)

LYAPUNOV_SOURCES = {
    "autonomous_driving": (
        ROOT
        / "results"
        / "safety"
        / "lyapunov-driving"
        / "sprint5-driving-lyapunov-summary.json"
    ),
    "robotics": (
        ROOT
        / "results"
        / "safety"
        / "lyapunov-robotics"
        / "sprint5-robotics-lyapunov-summary.json"
    ),
}

EXPECTED_SEEDS = (
    42,
    123,
    456,
)

EXPECTED_DOMAINS = (
    "autonomous_driving",
    "robotics",
)

EXPECTED_METHODS = (
    "none",
    "clipping",
    "lyapunov",
)


def _require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise AssertionError(message)


def _load_json(
    path: Path,
) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(f"Expected JSON object: {path}")

    return payload


def _load_csv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def _assert_close(
    actual: float,
    expected: float,
    *,
    message: str,
) -> None:
    _require(
        math.isclose(
            float(actual),
            float(expected),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ),
        message,
    )


def _record(
    payload: dict[str, Any],
    domain: str,
    method: str,
) -> dict[str, Any]:
    matches = [
        record
        for record in payload["records"]
        if (record["domain"] == domain and record["method"] == method)
    ]

    _require(
        len(matches) == 1,
        f"Expected one record for {domain}/{method}.",
    )

    return matches[0]


def _verify_protocol(
    payload: dict[str, Any],
) -> None:
    _require(
        payload["condition"] == "clean",
        "Safety condition must remain clean.",
    )

    _require(
        tuple(payload["required_seeds"]) == EXPECTED_SEEDS,
        "Seed protocol mismatch.",
    )

    _require(
        tuple(payload["domains"]) == EXPECTED_DOMAINS,
        "Domain protocol mismatch.",
    )

    _require(
        tuple(payload["safety_methods"]) == EXPECTED_METHODS,
        "Safety-method protocol mismatch.",
    )

    _require(
        payload["new_training_performed"] is False,
        "Unexpected training declaration.",
    )

    _require(
        payload["new_safety_evaluations_performed"] is False,
        "Unexpected new safety evaluation.",
    )

    _require(
        payload["new_robustness_experiments_performed"] is False,
        "Unexpected robustness experiment.",
    )

    semantics = payload["comparison_semantics"]

    _require(
        semantics["none"] == "REFERENCE",
        "NONE must remain REFERENCE.",
    )

    _require(
        semantics["clipping"] == "EVALUATED",
        "Clipping role mismatch.",
    )

    _require(
        semantics["lyapunov"] == "EVALUATED",
        "Lyapunov role mismatch.",
    )

    _require(
        semantics["binary_pass_fail_threshold_used"] is False,
        "Sprint 7.5 must not invent a binary threshold.",
    )


def _verify_matrix(
    payload: dict[str, Any],
) -> None:
    records = payload["records"]

    _require(
        len(records) == 6,
        "Expected six canonical records.",
    )

    observed = {
        (
            record["domain"],
            record["method"],
        )
        for record in records
    }

    expected = {
        (
            domain,
            method,
        )
        for domain in EXPECTED_DOMAINS
        for method in EXPECTED_METHODS
    }

    _require(
        observed == expected,
        "Six-record safety matrix mismatch.",
    )

    for record in records:
        _require(
            tuple(record["seeds"]) == EXPECTED_SEEDS,
            "Record seed set mismatch.",
        )


def _verify_against_frozen_sources(
    payload: dict[str, Any],
) -> None:
    baseline = _load_json(BASELINE_SOURCE)

    clipping = _load_json(CLIPPING_SOURCE)

    _require(
        baseline["condition"] == "clean",
        "Baseline source is not clean.",
    )

    _require(
        clipping["condition"] == "clean",
        "Clipping source is not clean.",
    )

    for domain in EXPECTED_DOMAINS:
        lyapunov = _load_json(LYAPUNOV_SOURCES[domain])

        _require(
            lyapunov["condition"] == "clean",
            "Lyapunov source is not clean.",
        )

        none_record = _record(
            payload,
            domain,
            "none",
        )

        clipping_record = _record(
            payload,
            domain,
            "clipping",
        )

        lyapunov_record = _record(
            payload,
            domain,
            "lyapunov",
        )

        baseline_domain = baseline["domains"][domain]

        _assert_close(
            none_record["violation_rate_mean"],
            baseline_domain["violation_step_rate"]["mean"],
            message=(f"{domain}/none violation mean mismatch."),
        )

        _assert_close(
            none_record["violation_rate_sample_std"],
            baseline_domain["violation_step_rate"]["sample_sd"],
            message=(f"{domain}/none violation SD mismatch."),
        )

        _assert_close(
            none_record["reward_mean"],
            baseline_domain["reward"]["mean_of_seed_means"],
            message=(f"{domain}/none reward mismatch."),
        )

        _assert_close(
            none_record["reward_sample_std"],
            baseline_domain["reward"]["sample_sd_of_seed_means"],
            message=(f"{domain}/none reward SD mismatch."),
        )

        _assert_close(
            none_record["success_rate_mean"],
            baseline_domain["success"]["mean_of_seed_rates"],
            message=(f"{domain}/none success mismatch."),
        )

        clipping_domain = clipping["domains"][domain]

        _assert_close(
            clipping_record["violation_rate_mean"],
            clipping_domain["clipping_violation_step_rate"]["mean"],
            message=(f"{domain}/clipping violation mismatch."),
        )

        _assert_close(
            clipping_record["reward_mean"],
            clipping_domain["clipping_reward"]["mean_of_seed_means"],
            message=(f"{domain}/clipping reward mismatch."),
        )

        _assert_close(
            clipping_record["success_rate_mean"],
            clipping_domain["clipping_success_rate"]["mean"],
            message=(f"{domain}/clipping success mismatch."),
        )

        _assert_close(
            clipping_record["intervention_rate_mean"],
            clipping_domain["intervention_rate"]["mean"],
            message=(f"{domain}/clipping intervention mismatch."),
        )

        lyapunov_summary = lyapunov["three_method_summary"]["lyapunov"]

        _assert_close(
            lyapunov_record["violation_rate_mean"],
            lyapunov_summary["violation_step_rate"]["mean"],
            message=(f"{domain}/lyapunov violation mismatch."),
        )

        _assert_close(
            lyapunov_record["reward_mean"],
            lyapunov_summary["reward"]["mean_of_seed_means"],
            message=(f"{domain}/lyapunov reward mismatch."),
        )

        _assert_close(
            lyapunov_record["success_rate_mean"],
            lyapunov_summary["success"]["mean_of_seed_rates"],
            message=(f"{domain}/lyapunov success mismatch."),
        )

        _assert_close(
            lyapunov_record["intervention_rate_mean"],
            lyapunov_summary["intervention_rate"]["mean"],
            message=(f"{domain}/lyapunov intervention mismatch."),
        )


def _verify_violation_calculations(
    payload: dict[str, Any],
) -> None:
    for domain in EXPECTED_DOMAINS:
        reference = _record(
            payload,
            domain,
            "none",
        )

        reference_rate = float(reference["violation_rate_mean"])

        _assert_close(
            reference["violation_absolute_difference_vs_none"],
            0.0,
            message=(f"{domain}/none absolute delta mismatch."),
        )

        _require(
            reference["violation_relative_reduction_percent_vs_none"] is None,
            "Reference relative reduction must be undefined.",
        )

        for method in (
            "clipping",
            "lyapunov",
        ):
            record = _record(
                payload,
                domain,
                method,
            )

            candidate = float(record["violation_rate_mean"])

            expected_absolute = candidate - reference_rate

            _assert_close(
                record["violation_absolute_difference_vs_none"],
                expected_absolute,
                message=(f"{domain}/{method} absolute delta mismatch."),
            )

            if reference_rate == 0.0:
                _require(
                    record["violation_relative_reduction_percent_vs_none"] is None,
                    "Zero reference must produce undefined reduction.",
                )
            else:
                expected_relative = (
                    100.0 * (reference_rate - candidate) / reference_rate
                )

                _assert_close(
                    record["violation_relative_reduction_percent_vs_none"],
                    expected_relative,
                    message=(f"{domain}/{method} " "relative reduction mismatch."),
                )


def _verify_missing_values(
    payload: dict[str, Any],
) -> None:
    for domain in EXPECTED_DOMAINS:
        reference = _record(
            payload,
            domain,
            "none",
        )

        _require(
            reference["intervention_rate_mean"] is None,
            "NONE intervention rate must remain undefined.",
        )

        _require(
            reference["intervention_rate_sample_std"] is None,
            "NONE intervention SD must remain undefined.",
        )


def _verify_zero_observed_semantics(
    payload: dict[str, Any],
) -> None:
    for domain in EXPECTED_DOMAINS:
        reference = _record(
            payload,
            domain,
            "none",
        )

        _require(
            reference["violation_rate_mean"] > 0.0,
            "Frozen NONE violation rate must remain nonzero.",
        )

        for method in (
            "clipping",
            "lyapunov",
        ):
            record = _record(
                payload,
                domain,
                method,
            )

            _assert_close(
                record["violation_rate_mean"],
                0.0,
                message=(
                    f"{domain}/{method} must retain " "zero observed violation rate."
                ),
            )

            _require(
                record["zero_observed_violations"] is True,
                "Zero-observed flag mismatch.",
            )


def _verify_mechanism(
    payload: dict[str, Any],
) -> None:
    observations = payload["lyapunov_mechanism_observations"]

    _require(
        len(observations) == 2,
        "Expected two mechanism observations.",
    )

    observed_domains = {observation["domain"] for observation in observations}

    _require(
        observed_domains == set(EXPECTED_DOMAINS),
        "Mechanism domains mismatch.",
    )

    for observation in observations:
        _require(
            observation["lyapunov_decrease_interventions_observed"] is False,
            "Clean principal runs did not observe " "LYAPUNOV_DECREASE interventions.",
        )

        _require(
            bool(observation["interpretation"].strip()),
            "Mechanism interpretation missing.",
        )


def _verify_provenance(
    payload: dict[str, Any],
) -> None:
    sources = payload["source_provenance"]

    _require(
        bool(sources),
        "Source provenance missing.",
    )

    for source in sources:
        _require(
            source.startswith("results/safety/"),
            "Unexpected provenance source.",
        )

        _require(
            "final-validation" not in source,
            "Generated output cannot be canonical input.",
        )

        _require(
            "gaussian-robustness" not in source,
            "Gaussian robustness contaminated Sprint 7.5.",
        )

        _require(
            "structured-state-robustness" not in source,
            "Structured-state robustness contaminated Sprint 7.5.",
        )

        _require(
            "action-robustness" not in source,
            "Action robustness contaminated Sprint 7.5.",
        )


def _verify_privileged_state_boundary(
    payload: dict[str, Any],
) -> None:
    limitation = payload["privileged_state_limitation"].lower()

    _require(
        "true simulator state" in limitation,
        "Privileged true-state limitation missing.",
    )

    _require(
        "perturbed observations" in limitation,
        "Policy-observation limitation missing.",
    )

    _require(
        "do not establish" in limitation,
        "Boundary language weakened.",
    )

    _require(
        "real-world sensing robustness" in limitation,
        "Real-world sensing limitation missing.",
    )


def _verify_claim_controls(
    payload: dict[str, Any],
) -> None:
    controls = payload["claim_controls"]

    expected_blocked = (
        "formal_stability_guarantee_allowed",
        "formal_lyapunov_stability_proof_allowed",
        "forward_invariance_guarantee_allowed",
        "worst_case_safety_guarantee_allowed",
        "iso_26262_certification_claim_allowed",
        "production_safety_claim_allowed",
        "real_world_safety_claim_allowed",
        "universal_safety_controller_claim_allowed",
        "quantum_safety_advantage_claim_allowed",
        "guaranteed_zero_violations_claim_allowed",
    )

    for key in expected_blocked:
        _require(
            controls[key] is False,
            f"Claim must remain blocked: {key}",
        )

    for domain in EXPECTED_DOMAINS:
        lyapunov = _record(
            payload,
            domain,
            "lyapunov",
        )

        _require(
            lyapunov["lyapunov_method_classification"] == "classical",
            "Lyapunov method must remain classical.",
        )

    conclusion = payload["scientific_conclusion"].lower()

    prohibited_positive_claims = (
        "guaranteed zero violations",
        "quantum lyapunov safety",
        "certified safe",
        "production safe",
        "formally guaranteed safe",
    )

    for phrase in prohibited_positive_claims:
        _require(
            phrase not in conclusion,
            f"Prohibited claim detected: {phrase}",
        )

    _require(
        "do not constitute" in conclusion,
        "Scientific conclusion must preserve bounded claims.",
    )


def _verify_csvs(
    payload: dict[str, Any],
    csv_rows: list[dict[str, str]],
    plot_rows: list[dict[str, str]],
) -> None:
    _require(
        len(csv_rows) == 6,
        "Safety CSV must contain six records.",
    )

    _require(
        len(plot_rows) == 6,
        "Plot dataset must contain six records.",
    )

    json_matrix = {
        (
            record["domain"],
            record["method"],
        )
        for record in payload["records"]
    }

    csv_matrix = {
        (
            row["domain"],
            row["method"],
        )
        for row in csv_rows
    }

    plot_matrix = {
        (
            row["domain"],
            row["method"],
        )
        for row in plot_rows
    }

    _require(
        json_matrix == csv_matrix == plot_matrix,
        "JSON/CSV/plot matrix mismatch.",
    )

    for row in csv_rows:
        if row["method"] == "none":
            _require(
                row["intervention_rate_mean"] == "",
                "Undefined NONE intervention rate " "must remain blank in CSV.",
            )


def _verify_markdown(
    markdown: str,
) -> None:
    required = (
        "## Autonomous Driving",
        "## Robotics",
        "## Lyapunov Mechanism Observation",
        "## Scientific Conclusion",
        "## Privileged-State Limitation",
        "## Claim Controls",
        "Formal stability guarantee: blocked",
        "Quantum-safety advantage claim: blocked",
        "Guaranteed-zero-violations claim: blocked",
        "classical",
        "LYAPUNOV_DECREASE intervention observed: no",
    )

    for text in required:
        _require(
            text in markdown,
            f"Markdown missing required text: {text}",
        )


def main() -> None:
    json_path = OUTPUT_DIR / "safety-ablation.json"

    csv_path = OUTPUT_DIR / "safety-ablation.csv"

    markdown_path = OUTPUT_DIR / "safety-ablation.md"

    plot_path = OUTPUT_DIR / "safety-ablation-plot-data.csv"

    for path in (
        json_path,
        csv_path,
        markdown_path,
        plot_path,
    ):
        _require(
            path.exists(),
            f"Missing generated artifact: {path}",
        )

    payload = _load_json(json_path)

    csv_rows = _load_csv(csv_path)

    plot_rows = _load_csv(plot_path)

    markdown = markdown_path.read_text(
        encoding="utf-8",
    )

    _verify_protocol(payload)

    _verify_matrix(payload)

    _verify_against_frozen_sources(payload)

    _verify_violation_calculations(payload)

    _verify_missing_values(payload)

    _verify_zero_observed_semantics(payload)

    _verify_mechanism(payload)

    _verify_provenance(payload)

    _verify_privileged_state_boundary(payload)

    _verify_claim_controls(payload)

    _verify_csvs(
        payload,
        csv_rows,
        plot_rows,
    )

    _verify_markdown(markdown)

    print("=" * 64)
    print(" SPRINT 7.5 - SAFETY ABLATION VERIFICATION")
    print("=" * 64)
    print()
    print("Protocol:                      PASS")
    print("2 domains:                     PASS")
    print("3 safety methods:              PASS")
    print("6-record matrix:               PASS")
    print("Seeds 42/123/456:              PASS")
    print("Mean/sample-SD semantics:      PASS")
    print("NONE reference handling:       PASS")
    print("Violation calculations:        PASS")
    print("Missing-value handling:        PASS")
    print("Zero-observed semantics:       PASS")
    print("Provenance:                    PASS")
    print("Privileged-state boundary:     PASS")
    print("Lyapunov classical label:      PASS")
    print("Mechanism interpretation:      PASS")
    print("Plot-data consistency:         PASS")
    print("Claim controls:                PASS")
    print()
    print("INDEPENDENT VERIFICATION: PASS")


if __name__ == "__main__":
    main()
