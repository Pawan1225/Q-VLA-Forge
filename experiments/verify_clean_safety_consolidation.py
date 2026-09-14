"""Independently verify Sprint 5.13B clean safety consolidation."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

SAFETY_ROOT = ROOT / "results" / "safety"

SUMMARY_PATH = SAFETY_ROOT / "consolidated" / "sprint5-clean-three-seed-summary.json"

CLIPPING_PATH = SAFETY_ROOT / "clipping" / "sprint5-clipping-safety-summary.json"

DRIVING_LYAPUNOV_PATH = (
    SAFETY_ROOT / "lyapunov-driving" / "sprint5-driving-lyapunov-summary.json"
)

ROBOTICS_LYAPUNOV_PATH = (
    SAFETY_ROOT / "lyapunov-robotics" / "sprint5-robotics-lyapunov-summary.json"
)

SEEDS = (
    42,
    123,
    456,
)

DOMAINS = (
    "autonomous_driving",
    "robotics",
)

METHODS = (
    "none",
    "clipping",
    "lyapunov",
)


def _load(
    path: Path,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(f"expected dict: {path}")

    return payload


def _check(
    condition: bool,
    label: str,
) -> None:
    if not condition:
        raise AssertionError(label)

    print(f"[PASS] {label}")


def _close(
    actual: float,
    expected: float,
    *,
    tol: float = 1e-12,
) -> bool:
    return abs(actual - expected) <= tol


def _summarize(
    values: list[float],
) -> tuple[float, float]:
    return (
        float(mean(values)),
        float(stdev(values)),
    )


def main() -> None:
    print("=" * 84)
    print(" Q-VLA FORGE - SPRINT 5.13B CLEAN SAFETY VERIFICATION")
    print("=" * 84)
    print()

    summary = _load(SUMMARY_PATH)

    clipping = _load(CLIPPING_PATH)

    driving_lyapunov = _load(DRIVING_LYAPUNOV_PATH)

    robotics_lyapunov = _load(ROBOTICS_LYAPUNOV_PATH)

    _check(
        summary["analysis_only"] is True,
        "analysis-only scope",
    )

    _check(
        summary["new_training"] is False,
        "no new training",
    )

    _check(
        summary["new_principal_runs"] is False,
        "no new principal runs",
    )

    seed_rows = summary["seed_rows"]

    _check(
        isinstance(
            seed_rows,
            list,
        )
        and len(seed_rows) == 18,
        "18 seed-level clean rows",
    )

    reconstructed: dict[
        tuple[
            str,
            str,
            int,
        ],
        dict[str, float],
    ] = {}

    clipping_rows = clipping["seed_level_comparisons"]

    if not isinstance(
        clipping_rows,
        list,
    ):
        raise TypeError("clipping seed comparisons must be a list")

    for raw in clipping_rows:
        if not isinstance(
            raw,
            dict,
        ):
            raise TypeError("clipping comparison row must be dict")

        domain = str(raw["domain"])

        seed = int(raw["principal_seed"])

        reconstructed[
            (
                domain,
                "none",
                seed,
            )
        ] = {
            "violation_step_rate": float(raw["none_violation_step_rate"]),
            "reward": float(raw["none_reward"]),
            "success_rate": float(raw["none_success_rate"]),
            "intervention_rate": 0.0,
            "mean_correction_l2": 0.0,
        }

        reconstructed[
            (
                domain,
                "clipping",
                seed,
            )
        ] = {
            "violation_step_rate": float(raw["clipping_violation_step_rate"]),
            "reward": float(raw["clipping_reward"]),
            "success_rate": float(raw["clipping_success_rate"]),
            "intervention_rate": float(raw["intervention_rate"]),
            "mean_correction_l2": float(raw["mean_action_correction_l2"]),
        }

    for payload in (
        driving_lyapunov,
        robotics_lyapunov,
    ):
        domain = str(payload["domain"])

        rows = payload["seed_level_comparisons"]

        if not isinstance(
            rows,
            list,
        ):
            raise TypeError("Lyapunov seed comparisons must be a list")

        for raw in rows:
            if not isinstance(
                raw,
                dict,
            ):
                raise TypeError("Lyapunov comparison row must be dict")

            seed = int(raw["principal_seed"])

            reconstructed[
                (
                    domain,
                    "lyapunov",
                    seed,
                )
            ] = {
                "violation_step_rate": float(raw["lyapunov_violation_rate"]),
                "reward": float(raw["lyapunov_reward"]),
                "success_rate": float(raw["lyapunov_success_rate"]),
                "intervention_rate": float(raw["lyapunov_intervention_rate"]),
                "mean_correction_l2": float(raw["lyapunov_mean_correction_l2"]),
            }

    _check(
        len(reconstructed) == 18,
        "18 independently reconstructed source cells",
    )

    for raw in seed_rows:
        if not isinstance(
            raw,
            dict,
        ):
            raise TypeError("summary seed row must be dict")

        key = (
            str(raw["domain"]),
            str(raw["method"]),
            int(raw["principal_seed"]),
        )

        source = reconstructed[key]

        for metric in (
            "violation_step_rate",
            "reward",
            "success_rate",
            "intervention_rate",
            "mean_correction_l2",
        ):
            _check(
                _close(
                    float(raw[metric]),
                    source[metric],
                ),
                (f"{key[0]}/{key[1]}/" f"seed-{key[2]} {metric}"),
            )

    summaries = summary["three_seed_summary"]

    for domain in DOMAINS:
        for method in METHODS:
            for metric in (
                "violation_step_rate",
                "reward",
                "success_rate",
                "intervention_rate",
                "mean_correction_l2",
            ):
                values = [
                    reconstructed[
                        (
                            domain,
                            method,
                            seed,
                        )
                    ][metric]
                    for seed in SEEDS
                ]

                expected_mean, expected_sd = _summarize(values)

                reported = summaries[domain][method][metric]

                _check(
                    _close(
                        float(reported["mean"]),
                        expected_mean,
                    ),
                    (f"{domain}/{method}/" f"{metric} mean"),
                )

                _check(
                    _close(
                        float(reported["sample_sd"]),
                        expected_sd,
                    ),
                    (f"{domain}/{method}/" f"{metric} sample SD"),
                )

    expected_headlines = {
        (
            "autonomous_driving",
            "none",
        ): 0.39084090909090907,
        (
            "autonomous_driving",
            "clipping",
        ): 0.0,
        (
            "autonomous_driving",
            "lyapunov",
        ): 0.0,
        (
            "robotics",
            "none",
        ): 0.014666666666666666,
        (
            "robotics",
            "clipping",
        ): 0.0,
        (
            "robotics",
            "lyapunov",
        ): 0.0,
    }

    for (
        domain,
        method,
    ), expected in expected_headlines.items():
        actual = float(summaries[domain][method]["violation_step_rate"]["mean"])

        _check(
            _close(
                actual,
                expected,
            ),
            f"{domain}/{method} clean headline",
        )

    effectiveness = summary["effectiveness"]

    for domain in DOMAINS:
        for method in (
            "clipping",
            "lyapunov",
        ):
            gate = effectiveness[domain][method]

            _check(
                gate["domain_mean_violation_reduction_pass"] is True,
                f"{domain}/{method} mean reduction gate",
            )

            _check(
                gate["seed_safety_requirements_pass"] is True,
                f"{domain}/{method} seed safety gate",
            )

            _check(
                gate["reward_requirement_pass"] is True,
                f"{domain}/{method} reward gate",
            )

            _check(
                gate["success_requirement_pass"] is True,
                f"{domain}/{method} success gate",
            )

            _check(
                gate["overall_pass"] is True,
                f"{domain}/{method} overall effectiveness",
            )

    limitation = summary["aggregate_only_metrics"]["note"]

    _check(
        ("not available per seed" in str(limitation)),
        "aggregate-only metric limitation recorded",
    )

    print()
    print("Clean seed reconstruction: PASS")
    print("Three-seed means: PASS")
    print("Sample SD: PASS")
    print("Effectiveness accounting: PASS")
    print("Reporting limitation: PASS")
    print("No scientific reruns: PASS")

    print()
    print("SPRINT 5.13B CLEAN SAFETY VERIFICATION: PASS")


if __name__ == "__main__":
    main()
