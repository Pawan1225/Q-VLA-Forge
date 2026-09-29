"""Independent verification for Sprint 7.4 QML ablation."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "results" / "final-validation" / "qml-ablation"

PPO_TARGETS_SOURCE = ROOT / "results" / "rl" / "ppo" / "sprint4-ppo-targets.json"

ABLATION_SOURCE = (
    ROOT / "results" / "rl" / "ablation" / "sprint4-classical-vs-qml-ablation.json"
)

EXPECTED_SEEDS = (
    42,
    123,
    456,
)

EXPECTED_DOMAINS = (
    "autonomous_driving",
    "robotics",
)

EXPECTED_POLICIES = (
    "ppo_mlp",
    "ppo_pqc",
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

    _require(
        isinstance(
            payload,
            dict,
        ),
        f"Expected object: {path}",
    )

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


def _verify_protocol(
    payload: dict[str, Any],
) -> None:
    _require(
        tuple(payload["required_seeds"]) == EXPECTED_SEEDS,
        "Seed protocol mismatch.",
    )

    protocol = payload["protocol"]

    _require(
        tuple(protocol["domains"]) == EXPECTED_DOMAINS,
        "Domain protocol mismatch.",
    )

    _require(
        tuple(protocol["primary_policies"]) == EXPECTED_POLICIES,
        "Policy protocol mismatch.",
    )

    _require(
        protocol["primary_metric"] == "environment_steps_to_target",
        "Primary metric changed.",
    )

    _require(
        protocol["execution_backend"] == "simulator",
        "Execution backend mismatch.",
    )

    _require(
        protocol["quantum_hardware_used"] is False,
        "Quantum hardware must remain false.",
    )

    _require(
        protocol["maximum_environment_steps_is_not_convergence"] is True,
        "Budget/convergence distinction lost.",
    )


def _verify_primary_matrix(
    payload: dict[str, Any],
) -> None:
    records = payload["primary_records"]

    _require(
        len(records) == 12,
        "Expected twelve primary records.",
    )

    observed = {
        (
            record["domain"],
            record["policy"],
            int(record["seed"]),
        )
        for record in records
    }

    expected = {
        (
            domain,
            policy,
            seed,
        )
        for domain in EXPECTED_DOMAINS
        for policy in EXPECTED_POLICIES
        for seed in EXPECTED_SEEDS
    }

    _require(
        observed == expected,
        "Primary matrix mismatch.",
    )


def _verify_frozen_targets(
    payload: dict[str, Any],
) -> None:
    """Verify final records against original frozen PPO targets."""

    targets = _load_json(PPO_TARGETS_SOURCE)

    _require(
        targets["qml_results_seen"] is False,
        "Targets were not frozen independently of QML.",
    )

    for domain in EXPECTED_DOMAINS:
        domain_targets = targets["domains"][domain]

        _require(
            domain_targets["targets_frozen_before_qml"] is True,
            f"{domain}: targets not frozen before QML.",
        )

        for target in domain_targets["seeds"]:
            seed = int(target["seed"])

            matches = [
                record
                for record in payload["primary_records"]
                if record["domain"] == domain
                and record["policy"] == "ppo_mlp"
                and int(record["seed"]) == seed
            ]

            _require(
                len(matches) == 1,
                "Missing PPO reference record.",
            )

            record = matches[0]

            _require(
                record["target_reached"] is True,
                "Frozen PPO target must be reached.",
            )

            _require(
                int(record["environment_steps_to_target"])
                == int(target["ppo_environment_steps_to_target"]),
                "PPO steps-to-target differs from frozen target artifact.",
            )


def _verify_non_attainment(
    payload: dict[str, Any],
) -> None:
    qml = [
        record for record in payload["primary_records"] if record["policy"] == "ppo_pqc"
    ]

    _require(
        len(qml) == 6,
        "Expected six QML records.",
    )

    for record in qml:
        _require(
            record["target_reached"] is False,
            "QML target reach changed.",
        )

        _require(
            record["environment_steps_to_target"] is None,
            ("QML non-attainment must keep " "steps_to_target=null."),
        )

        _require(
            record["episodes_to_target"] is None,
            ("QML non-attainment must keep " "episodes_to_target=null."),
        )

        _require(
            record["environment_steps_to_target"] != 20_000,
            ("Maximum budget must not be substituted " "for convergence."),
        )


def _verify_target_counts(
    payload: dict[str, Any],
) -> None:
    summary = payload["cross_domain_summary"]

    _require(
        summary["classical_ppo_target_reach"]
        == {
            "reached": 6,
            "available": 6,
        },
        "Classical PPO target count mismatch.",
    )

    _require(
        summary["qml_pqc_target_reach"]
        == {
            "reached": 0,
            "available": 6,
        },
        "QML target count mismatch.",
    )

    _require(
        summary["matched_classical_target_reach"]
        == {
            "reached": 1,
            "available": 6,
        },
        "Matched-classical target count mismatch.",
    )


def _verify_matched_control(
    payload: dict[str, Any],
) -> None:
    controls = payload["matched_classical_controls"]

    _require(
        len(controls) == 6,
        "Expected six matched controls.",
    )

    original = _load_json(ABLATION_SOURCE)

    _require(
        original["primary_target_reach"]["matched_classical"]["reached"] == 1,
        "Original matched-classical reach mismatch.",
    )

    _require(
        original["primary_target_reach"]["qml"]["reached"] == 0,
        "Original QML reach mismatch.",
    )

    for control in controls:
        if not control["target_reached"]:
            _require(
                control["environment_steps_to_target"] is None,
                ("Failed matched control must have " "null steps-to-target."),
            )


def _verify_compactness(
    payload: dict[str, Any],
) -> None:
    compactness = {record["domain"]: record for record in payload["compactness"]}

    driving = compactness["autonomous_driving"]

    robotics = compactness["robotics"]

    _require(
        driving["classical_actor_parameters"] == 1318,
        "Driving PPO parameter mismatch.",
    )

    _require(
        driving["qml_actor_parameters"] == 54,
        "Driving PQC parameter mismatch.",
    )

    _require(
        math.isclose(
            float(driving["parameter_reduction_percent"]),
            100.0 * (1.0 - 54.0 / 1318.0),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ),
        "Driving compactness calculation mismatch.",
    )

    _require(
        robotics["classical_actor_parameters"] == 1382,
        "Robotics PPO parameter mismatch.",
    )

    _require(
        robotics["qml_actor_parameters"] == 62,
        "Robotics PQC parameter mismatch.",
    )

    _require(
        math.isclose(
            float(robotics["parameter_reduction_percent"]),
            100.0 * (1.0 - 62.0 / 1382.0),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ),
        "Robotics compactness calculation mismatch.",
    )

    _require(
        driving["statistical"] is False,
        "Parameter count must be deterministic.",
    )

    _require(
        robotics["statistical"] is False,
        "Parameter count must be deterministic.",
    )


def _verify_provenance(
    payload: dict[str, Any],
) -> None:
    sources = payload["source_provenance"]

    _require(
        bool(sources),
        "Missing provenance.",
    )

    for source in sources:
        _require(
            source.startswith("results/rl/"),
            "Non-RL source detected.",
        )

        _require(
            "final-validation" not in source,
            "Generated evidence used as canonical input.",
        )


def _verify_claim_controls(
    payload: dict[str, Any],
) -> None:
    controls = payload["claim_controls"]

    required_blocked = (
        "quantum_advantage_claim_allowed",
        "quantum_speedup_claim_allowed",
        "qml_sample_efficiency_advantage_claim_allowed",
        "qml_performance_superiority_claim_allowed",
        "qml_convergence_superiority_claim_allowed",
        "qpu_advantage_claim_allowed",
        "production_vla_superiority_claim_allowed",
    )

    for key in required_blocked:
        _require(
            controls[key] is False,
            f"Claim must remain blocked: {key}",
        )

    conclusion = payload["scientific_conclusion"].lower()

    _require(
        "compact" not in conclusion or "advantage" in conclusion,
        "Malformed scientific conclusion.",
    )

    _require(
        "did not demonstrate" in conclusion,
        "Negative QML result must remain explicit.",
    )


def _verify_csvs(
    payload: dict[str, Any],
    primary_rows: list[dict[str, str]],
    plot_rows: list[dict[str, str]],
) -> None:
    _require(
        len(primary_rows) == 12,
        "Primary CSV must contain 12 rows.",
    )

    _require(
        len(plot_rows) == 18,
        ("Plot data must contain " "12 primary + 6 control rows."),
    )

    qml_plot_rows = [row for row in plot_rows if row["policy"] == "ppo_pqc"]

    _require(
        len(qml_plot_rows) == 6,
        "Plot data missing QML rows.",
    )

    for row in qml_plot_rows:
        _require(
            row["environment_steps_to_target"] == "",
            ("QML plot data must preserve missing " "steps-to-target as blank/null."),
        )

    _require(
        len(payload["primary_records"]) == len(primary_rows),
        "JSON/CSV primary row count mismatch.",
    )


def _verify_markdown(
    markdown: str,
) -> None:
    required = (
        "## Autonomous Driving",
        "## Robotics",
        "## Cross-Domain Result",
        "## Scientific Conclusion",
        "## Claim Controls",
        "Quantum advantage: blocked",
        "Quantum speedup: blocked",
    )

    for text in required:
        _require(
            text in markdown,
            f"Markdown missing: {text}",
        )

    _require(
        "20,000-step training budget is not treated" in markdown,
        "Markdown lost non-attainment boundary.",
    )


def main() -> None:
    json_path = OUTPUT_DIR / "qml-ablation.json"

    csv_path = OUTPUT_DIR / "qml-ablation.csv"

    markdown_path = OUTPUT_DIR / "qml-ablation.md"

    plot_path = OUTPUT_DIR / "qml-ablation-plot-data.csv"

    for path in (
        json_path,
        csv_path,
        markdown_path,
        plot_path,
    ):
        _require(
            path.exists(),
            f"Missing artifact: {path}",
        )

    payload = _load_json(json_path)

    primary_rows = _load_csv(csv_path)

    plot_rows = _load_csv(plot_path)

    markdown = markdown_path.read_text(
        encoding="utf-8",
    )

    _verify_protocol(payload)

    _verify_primary_matrix(payload)

    _verify_frozen_targets(payload)

    _verify_non_attainment(payload)

    _verify_target_counts(payload)

    _verify_matched_control(payload)

    _verify_compactness(payload)

    _verify_provenance(payload)

    _verify_claim_controls(payload)

    _verify_csvs(
        payload,
        primary_rows,
        plot_rows,
    )

    _verify_markdown(markdown)

    print("=" * 64)
    print(" SPRINT 7.4 - QML ABLATION VERIFICATION")
    print("=" * 64)
    print()
    print("Protocol:                    PASS")
    print("Two domains:                 PASS")
    print("Three seeds:                 PASS")
    print("PPO/MLP evidence:            PASS")
    print("PQC/QML evidence:            PASS")
    print("Target definition frozen:    PASS")
    print("Non-attainment semantics:    PASS")
    print("Parameter accounting:        PASS")
    print("Matched-classical control:   PASS")
    print("Simulator declaration:       PASS")
    print("Provenance:                  PASS")
    print("Plot data:                   PASS")
    print("Claim controls:              PASS")
    print()
    print("INDEPENDENT VERIFICATION: PASS")


if __name__ == "__main__":
    main()
