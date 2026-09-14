"""Build Sprint 5.13B clean three-seed safety evidence."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.clean_safety_consolidation import (
    PRINCIPAL_SEEDS,
    MetricSummary,
    SeedCleanResult,
    effectiveness_gate,
    summarize_results,
)

ROOT = Path(__file__).resolve().parents[1]

SAFETY_ROOT = ROOT / "results" / "safety"

OUTPUT_ROOT = SAFETY_ROOT / "consolidated"

OUTPUT_JSON = OUTPUT_ROOT / "sprint5-clean-three-seed-summary.json"

OUTPUT_CSV = OUTPUT_ROOT / "sprint5-clean-three-seed-summary.csv"

OUTPUT_MD = OUTPUT_ROOT / "sprint5-clean-three-seed-summary.md"


def _load(
    path: Path,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(f"expected object: {path}")

    return payload


def _float(
    row: dict[str, Any],
    key: str,
) -> float:
    return float(row[key])


def _int(
    row: dict[str, Any],
    key: str,
) -> int:
    return int(row[key])


def _build_none_and_clipping(
    clipping: dict[str, Any],
) -> tuple[
    dict[str, list[SeedCleanResult]],
    dict[str, list[SeedCleanResult]],
]:
    none: dict[str, list[SeedCleanResult]] = {
        "autonomous_driving": [],
        "robotics": [],
    }

    clip: dict[str, list[SeedCleanResult]] = {
        "autonomous_driving": [],
        "robotics": [],
    }

    rows = clipping["seed_level_comparisons"]

    if not isinstance(
        rows,
        list,
    ):
        raise TypeError("seed_level_comparisons must be a list")

    for raw in rows:
        if not isinstance(
            raw,
            dict,
        ):
            raise TypeError("comparison row must be a dict")

        row: dict[str, Any] = raw

        domain = str(row["domain"])

        seed = _int(
            row,
            "principal_seed",
        )

        none[domain].append(
            SeedCleanResult(
                domain=domain,
                method="none",
                principal_seed=seed,
                violation_step_rate=_float(
                    row,
                    "none_violation_step_rate",
                ),
                reward=_float(
                    row,
                    "none_reward",
                ),
                success_rate=_float(
                    row,
                    "none_success_rate",
                ),
                intervention_rate=0.0,
                mean_correction_l2=0.0,
            )
        )

        clip[domain].append(
            SeedCleanResult(
                domain=domain,
                method="clipping",
                principal_seed=seed,
                violation_step_rate=_float(
                    row,
                    "clipping_violation_step_rate",
                ),
                reward=_float(
                    row,
                    "clipping_reward",
                ),
                success_rate=_float(
                    row,
                    "clipping_success_rate",
                ),
                intervention_rate=_float(
                    row,
                    "intervention_rate",
                ),
                mean_correction_l2=_float(
                    row,
                    "mean_action_correction_l2",
                ),
            )
        )

    return (
        none,
        clip,
    )


def _build_lyapunov(
    payload: dict[str, Any],
) -> list[SeedCleanResult]:
    domain = str(payload["domain"])

    rows = payload["seed_level_comparisons"]

    if not isinstance(
        rows,
        list,
    ):
        raise TypeError("Lyapunov seed rows must be a list")

    results: list[SeedCleanResult] = []

    for raw in rows:
        if not isinstance(
            raw,
            dict,
        ):
            raise TypeError("Lyapunov row must be a dict")

        row: dict[str, Any] = raw

        results.append(
            SeedCleanResult(
                domain=domain,
                method="lyapunov",
                principal_seed=_int(
                    row,
                    "principal_seed",
                ),
                violation_step_rate=_float(
                    row,
                    "lyapunov_violation_rate",
                ),
                reward=_float(
                    row,
                    "lyapunov_reward",
                ),
                success_rate=_float(
                    row,
                    "lyapunov_success_rate",
                ),
                intervention_rate=_float(
                    row,
                    "lyapunov_intervention_rate",
                ),
                mean_correction_l2=_float(
                    row,
                    "lyapunov_mean_correction_l2",
                ),
            )
        )

    return results


def _metric_to_dict(
    metric: MetricSummary,
) -> dict[str, Any]:
    return asdict(metric)


def main() -> None:
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    clipping = _load(SAFETY_ROOT / "clipping" / "sprint5-clipping-safety-summary.json")

    driving_lyapunov = _load(
        SAFETY_ROOT / "lyapunov-driving" / "sprint5-driving-lyapunov-summary.json"
    )

    robotics_lyapunov = _load(
        SAFETY_ROOT / "lyapunov-robotics" / "sprint5-robotics-lyapunov-summary.json"
    )

    none, clip = _build_none_and_clipping(clipping)

    lyapunov = {
        "autonomous_driving": _build_lyapunov(driving_lyapunov),
        "robotics": _build_lyapunov(robotics_lyapunov),
    }

    methods = {
        "none": none,
        "clipping": clip,
        "lyapunov": lyapunov,
    }

    seed_rows: list[dict[str, Any]] = []

    summaries: dict[str, dict[str, dict[str, Any]]] = {}

    for domain in (
        "autonomous_driving",
        "robotics",
    ):
        summaries[domain] = {}

        for method in (
            "none",
            "clipping",
            "lyapunov",
        ):
            results = methods[method][domain]

            results.sort(key=lambda item: item.principal_seed)

            if [result.principal_seed for result in results] != list(PRINCIPAL_SEEDS):
                raise RuntimeError(f"incomplete seeds: {domain}/{method}")

            for seed_result in results:
                seed_rows.append(asdict(seed_result))

            metric_summary = summarize_results(results)

            summaries[domain][method] = {
                key: _metric_to_dict(value)
                for (
                    key,
                    value,
                ) in metric_summary.items()
            }

    effectiveness: dict[str, dict[str, dict[str, object]]] = {}

    for domain in (
        "autonomous_driving",
        "robotics",
    ):
        effectiveness[domain] = {}

        none_rows = {result.principal_seed: result for result in none[domain]}

        for method in (
            "clipping",
            "lyapunov",
        ):
            method_rows = {
                result.principal_seed: result for result in methods[method][domain]
            }

            effectiveness[domain][method] = effectiveness_gate(
                none_violation_by_seed={
                    seed: none_rows[seed].violation_step_rate
                    for seed in PRINCIPAL_SEEDS
                },
                comparison_violation_by_seed={
                    seed: method_rows[seed].violation_step_rate
                    for seed in PRINCIPAL_SEEDS
                },
                none_reward_mean=(summaries[domain]["none"]["reward"]["mean"]),
                comparison_reward_mean=(summaries[domain][method]["reward"]["mean"]),
                none_success_mean=(summaries[domain]["none"]["success_rate"]["mean"]),
                comparison_success_mean=(
                    summaries[domain][method]["success_rate"]["mean"]
                ),
            )

    aggregate_only = {
        "autonomous_driving": {
            "none": {
                "critical_violation_step_rate": clipping["seed_level_comparisons"][
                    0
                ].get("none_critical_violation_step_rate")
            }
        },
        "note": (
            "Critical and constraint violation rates "
            "are not available per seed for every "
            "clean method in the frozen summary "
            "artifacts and therefore are not "
            "fabricated in the three-seed table."
        ),
    }

    payload = {
        "sprint": "5.13B",
        "artifact": "clean-three-seed-safety-reconstruction",
        "analysis_only": True,
        "new_training": False,
        "new_principal_runs": False,
        "principal_seeds": list(PRINCIPAL_SEEDS),
        "domains": [
            "autonomous_driving",
            "robotics",
        ],
        "methods": [
            "none",
            "clipping",
            "lyapunov",
        ],
        "seed_rows": seed_rows,
        "three_seed_summary": summaries,
        "effectiveness": effectiveness,
        "aggregate_only_metrics": aggregate_only,
    }

    OUTPUT_JSON.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "domain",
                "method",
                "principal_seed",
                "violation_step_rate",
                "reward",
                "success_rate",
                "intervention_rate",
                "mean_correction_l2",
            ],
        )

        writer.writeheader()
        writer.writerows(seed_rows)

    lines = [
        ("# Sprint 5.13B — Clean Three-Seed " "Safety Reconstruction"),
        "",
        ("Analysis only. No new policy or " "environment execution."),
        "",
        (
            "| Domain | Method | Violation-step rate | "
            "Reward | Success | Intervention |"
        ),
        "|---|---|---:|---:|---:|---:|",
    ]

    for domain in (
        "autonomous_driving",
        "robotics",
    ):
        for method in (
            "none",
            "clipping",
            "lyapunov",
        ):
            summary_row = summaries[domain][method]

            lines.append(
                "| "
                + domain
                + " | "
                + method
                + " | "
                + (
                    f"{summary_row['violation_step_rate']['mean']:.6f} "
                    f"± "
                    f"{summary_row['violation_step_rate']['sample_sd']:.6f}"
                )
                + " | "
                + (
                    f"{summary_row['reward']['mean']:.6f} "
                    f"± "
                    f"{summary_row['reward']['sample_sd']:.6f}"
                )
                + " | "
                + (
                    f"{summary_row['success_rate']['mean']:.6f} "
                    f"± "
                    f"{summary_row['success_rate']['sample_sd']:.6f}"
                )
                + " | "
                + (
                    f"{summary_row['intervention_rate']['mean']:.6f} "
                    f"± "
                    f"{summary_row['intervention_rate']['sample_sd']:.6f}"
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Effectiveness",
            "",
        ]
    )

    for domain in (
        "autonomous_driving",
        "robotics",
    ):
        for method in (
            "clipping",
            "lyapunov",
        ):
            gate = effectiveness[domain][method]

            lines.append(
                f"- {domain} / {method}: "
                f"{'SUPPORTED' if gate['overall_pass'] else 'NOT SUPPORTED'}"
            )

    lines.extend(
        [
            "",
            "## Reporting limitation",
            "",
            (
                "Critical and constraint violation "
                "rates are not exposed per seed for "
                "every clean method by the frozen "
                "summary artifacts. They are therefore "
                "not fabricated in this three-seed "
                "reconstruction."
            ),
            "",
        ]
    )

    OUTPUT_MD.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("=" * 76)
    print(" SPRINT 5.13B CLEAN THREE-SEED RECONSTRUCTION")
    print("=" * 76)
    print()

    print(
        "Seed rows:",
        len(seed_rows),
        "/ 18",
    )

    for domain in (
        "autonomous_driving",
        "robotics",
    ):
        print()
        print(domain)

        for method in (
            "none",
            "clipping",
            "lyapunov",
        ):
            summary_row = summaries[domain][method]

            print(
                f"  {method:10s}"
                f" violation="
                f"{summary_row['violation_step_rate']['mean']:.9f}"
                f" ± "
                f"{summary_row['violation_step_rate']['sample_sd']:.9f}"
                f" reward="
                f"{summary_row['reward']['mean']:.9f}"
                f" success="
                f"{summary_row['success_rate']['mean']:.9f}"
            )

    print()
    print("Effectiveness gates:")

    for domain in (
        "autonomous_driving",
        "robotics",
    ):
        for method in (
            "clipping",
            "lyapunov",
        ):
            status = effectiveness[domain][method]["overall_pass"]

            print(
                f"  {domain}/{method}:",
                "PASS" if status else "FAIL",
            )

    print()
    print("No new training: PASS")
    print("No new principal runs: PASS")
    print()
    print("SPRINT 5.13B CLEAN RECONSTRUCTION: PASS")


if __name__ == "__main__":
    main()
