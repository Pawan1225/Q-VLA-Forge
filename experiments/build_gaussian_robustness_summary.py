"""Build the frozen Sprint 5.10 Gaussian robustness summary."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

RUNS = ROOT / "results" / "safety" / "gaussian-robustness" / "runs"

OUTPUT_ROOT = ROOT / "results" / "safety" / "gaussian-robustness"

DOMAINS = (
    "autonomous_driving",
    "robotics",
)

METHODS = (
    "none",
    "clipping",
    "lyapunov",
)

SEEDS = (
    42,
    123,
    456,
)

SIGMAS = (
    0.0,
    0.01,
    0.05,
    0.10,
)


def _sample_sd(
    values: list[float],
) -> float:
    if len(values) <= 1:
        return 0.0

    return float(stdev(values))


def _finite_float(
    value: Any,
    *,
    default: float = 0.0,
) -> float:
    try:
        result = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return default

    if not math.isfinite(result):
        return default

    return result


def _first(
    mapping: dict[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]

    return default


def _find_clean_run(
    *,
    domain: str,
    method: str,
    seed: int,
) -> Path:
    if method == "none":
        directory = ROOT / "results" / "safety" / "baseline" / "runs"

    elif method == "clipping":
        directory = ROOT / "results" / "safety" / "clipping" / "runs"

    elif method == "lyapunov" and domain == "autonomous_driving":
        directory = ROOT / "results" / "safety" / "lyapunov-driving" / "runs"

    elif method == "lyapunov" and domain == "robotics":
        directory = ROOT / "results" / "safety" / "lyapunov-robotics" / "runs"

    else:
        raise ValueError("unsupported clean reference")

    candidates = sorted(directory.glob(f"*seed-{seed}.json"))

    if not candidates:
        raise FileNotFoundError(
            f"clean run not found: " f"{domain} {method} seed {seed}"
        )

    domain_tokens = {
        "autonomous_driving": (
            "autonomous_driving",
            "autonomous-driving",
        ),
        "robotics": ("robotics",),
    }

    matches = [
        path
        for path in candidates
        if any(token in path.name for token in domain_tokens[domain])
    ]

    if len(matches) == 1:
        return matches[0]

    if len(candidates) == 1 and method == "lyapunov":
        return candidates[0]

    raise RuntimeError(
        "ambiguous clean reference: " f"{domain} {method} seed {seed}: " f"{candidates}"
    )


def _summary_mapping(
    payload: dict[str, Any],
) -> dict[str, Any]:
    for key in (
        "seed_summary",
        "summary",
    ):
        value = payload.get(key)

        if isinstance(
            value,
            dict,
        ):
            return value

    raise KeyError("clean artifact contains no seed summary")


def _normalize_cell(
    *,
    domain: str,
    method: str,
    seed: int,
    sigma: float,
    summary: dict[str, Any],
    source: str,
    checkpoint_sha256: str | None,
) -> dict[str, Any]:
    violation = _finite_float(
        _first(
            summary,
            "executed_violation_step_rate",
            "violation_step_rate",
        )
    )

    constraint = _finite_float(
        _first(
            summary,
            "executed_constraint_violation_rate",
            "constraint_violation_rate",
        )
    )

    return {
        "domain": domain,
        "method": method,
        "principal_seed": seed,
        "sigma": sigma,
        "source": source,
        "checkpoint_sha256": (checkpoint_sha256),
        "violation_step_rate": violation,
        "critical_violation_step_rate": (
            _finite_float(summary.get("critical_violation_step_rate"))
        ),
        "constraint_violation_rate": constraint,
        "reward": _finite_float(summary.get("mean_reward")),
        "success_rate": _finite_float(summary.get("success_rate")),
        "mean_episode_length": (_finite_float(summary.get("mean_episode_length"))),
        "intervention_rate": (_finite_float(summary.get("intervention_rate"))),
        "mean_action_correction_l2": (
            _finite_float(summary.get("mean_action_correction_l2"))
        ),
        "max_action_correction_l2": (
            _finite_float(summary.get("max_action_correction_l2"))
        ),
        "mean_noise_l2": (
            _finite_float(summary.get("mean_noise_l2")) if sigma > 0.0 else 0.0
        ),
        "max_noise_l2": (
            _finite_float(summary.get("max_noise_l2")) if sigma > 0.0 else 0.0
        ),
        "strict_lyapunov_decrease_rate": (
            _finite_float(summary.get("strict_lyapunov_decrease_rate"))
        ),
        "lyapunov_nonincrease_rate": (
            _finite_float(summary.get("lyapunov_nonincrease_rate"))
        ),
        "selected_lower_than_proposed_rate": (
            _finite_float(summary.get("selected_lower_than_proposed_rate"))
        ),
        "emergency_fallback_rate": (
            _finite_float(summary.get("emergency_fallback_rate"))
        ),
        "steps_object_grasped": int(
            summary.get(
                "steps_object_grasped",
                0,
            )
        ),
        "steps_object_not_grasped": int(
            summary.get(
                "steps_object_not_grasped",
                0,
            )
        ),
        "category_violation_rates": dict(
            summary.get(
                "category_violation_rates",
                {},
            )
        ),
        "intervention_reason_rates": dict(
            summary.get(
                "intervention_reason_rates",
                {},
            )
        ),
        "selected_candidate_source_rates": dict(
            summary.get(
                "selected_candidate_source_rates",
                {},
            )
        ),
        "intervened_candidate_source_rates": dict(
            summary.get(
                "intervened_candidate_source_rates",
                {},
            )
        ),
    }


def _load_clean_cells() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []

    for domain in DOMAINS:
        for method in METHODS:
            for seed in SEEDS:
                path = _find_clean_run(
                    domain=domain,
                    method=method,
                    seed=seed,
                )

                payload = json.loads(path.read_text(encoding="utf-8"))

                summary = _summary_mapping(payload)

                checkpoint_sha = payload.get("checkpoint_sha256")

                cells.append(
                    _normalize_cell(
                        domain=domain,
                        method=method,
                        seed=seed,
                        sigma=0.0,
                        summary=summary,
                        source=(str(path.relative_to(ROOT))),
                        checkpoint_sha256=(
                            str(checkpoint_sha) if checkpoint_sha is not None else None
                        ),
                    )
                )

    return cells


def _load_noisy_cells() -> tuple[
    list[dict[str, Any]],
    Counter[str],
    int,
    int,
    int,
]:
    files = sorted(RUNS.glob("*.json"))

    if len(files) != 54:
        raise RuntimeError(f"expected 54 noisy files, " f"found {len(files)}")

    cells: list[dict[str, Any]] = []

    reasons: Counter[str] = Counter()

    strict_decrease_steps = 0
    grasped_steps = 0
    total_steps = 0

    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))

        summary = payload["seed_summary"]

        cells.append(
            _normalize_cell(
                domain=str(payload["domain"]),
                method=str(payload["method"]),
                seed=int(payload["principal_seed"]),
                sigma=float(payload["sigma"]),
                summary=summary,
                source=str(path.relative_to(ROOT)),
                checkpoint_sha256=str(payload["checkpoint_sha256"]),
            )
        )

        if payload["method"] == "lyapunov":
            for episode in payload["episodes"]:
                reasons.update(
                    {
                        str(key): int(value)
                        for key, value in episode["intervention_reason_counts"].items()
                    }
                )

                strict_decrease_steps += int(episode["strict_lyapunov_decrease_count"])

                grasped_steps += int(episode["steps_object_grasped"])

                total_steps += int(episode["episode_length"])

    return (
        cells,
        reasons,
        strict_decrease_steps,
        grasped_steps,
        total_steps,
    )


def _add_clean_deltas(
    cells: list[dict[str, Any]],
) -> None:
    clean_lookup = {
        (
            cell["domain"],
            cell["method"],
            cell["principal_seed"],
        ): cell
        for cell in cells
        if cell["sigma"] == 0.0
    }

    for cell in cells:
        clean = clean_lookup[
            (
                cell["domain"],
                cell["method"],
                cell["principal_seed"],
            )
        ]

        cell["violation_delta_from_clean"] = (
            cell["violation_step_rate"] - clean["violation_step_rate"]
        )

        cell["critical_delta_from_clean"] = (
            cell["critical_violation_step_rate"] - clean["critical_violation_step_rate"]
        )

        cell["constraint_delta_from_clean"] = (
            cell["constraint_violation_rate"] - clean["constraint_violation_rate"]
        )

        cell["reward_delta_from_clean"] = cell["reward"] - clean["reward"]

        cell["success_delta_from_clean"] = cell["success_rate"] - clean["success_rate"]

        if clean["violation_step_rate"] > 0.0:
            cell["relative_violation_degradation"] = (
                cell["violation_delta_from_clean"] / clean["violation_step_rate"]
            )
        else:
            cell["relative_violation_degradation"] = None


def _aggregate_cells(
    cells: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[
        tuple[str, str, float],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for cell in cells:
        groups[
            (
                str(cell["domain"]),
                str(cell["method"]),
                float(cell["sigma"]),
            )
        ].append(cell)

    metric_names = (
        "violation_step_rate",
        "critical_violation_step_rate",
        "constraint_violation_rate",
        "reward",
        "success_rate",
        "mean_episode_length",
        "intervention_rate",
        "mean_action_correction_l2",
        "mean_noise_l2",
        "violation_delta_from_clean",
        "critical_delta_from_clean",
        "constraint_delta_from_clean",
        "reward_delta_from_clean",
        "success_delta_from_clean",
        "strict_lyapunov_decrease_rate",
        "lyapunov_nonincrease_rate",
        "selected_lower_than_proposed_rate",
        "emergency_fallback_rate",
    )

    output: list[dict[str, Any]] = []

    for (
        domain,
        method,
        sigma,
    ), group in sorted(groups.items()):
        if len(group) != 3:
            raise RuntimeError(
                "aggregate group does not " "contain three principal seeds"
            )

        row: dict[str, Any] = {
            "domain": domain,
            "method": method,
            "sigma": sigma,
            "principal_seed_count": 3,
        }

        for metric in metric_names:
            values = [_finite_float(cell[metric]) for cell in group]

            row[metric] = {
                "mean": float(mean(values)),
                "sample_sd": (_sample_sd(values)),
            }

        row["max_action_correction_l2"] = max(
            _finite_float(cell["max_action_correction_l2"]) for cell in group
        )

        row["steps_object_grasped"] = sum(
            int(cell["steps_object_grasped"]) for cell in group
        )

        output.append(row)

    return output


def _write_csv(
    cells: list[dict[str, Any]],
    path: Path,
) -> None:
    columns = (
        "domain",
        "method",
        "principal_seed",
        "sigma",
        "violation_step_rate",
        "violation_delta_from_clean",
        "critical_violation_step_rate",
        "critical_delta_from_clean",
        "constraint_violation_rate",
        "constraint_delta_from_clean",
        "reward",
        "reward_delta_from_clean",
        "success_rate",
        "success_delta_from_clean",
        "intervention_rate",
        "mean_action_correction_l2",
        "max_action_correction_l2",
        "mean_noise_l2",
        "strict_lyapunov_decrease_rate",
        "lyapunov_nonincrease_rate",
        "selected_lower_than_proposed_rate",
        "emergency_fallback_rate",
        "steps_object_grasped",
        "source",
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
        )

        writer.writeheader()

        for cell in cells:
            writer.writerow({key: cell.get(key) for key in columns})


def _write_markdown(
    aggregates: list[dict[str, Any]],
    findings: dict[str, Any],
    path: Path,
) -> None:
    lines = [
        "# Sprint 5.10 — Gaussian Observation-Noise Robustness",
        "",
        "## Protocol",
        "",
        "- Domains: autonomous driving and robotics",
        "- Methods: NONE, CLIPPING, LYAPUNOV",
        "- Principal seeds: 42, 123, 456",
        "- Gaussian sigma: 0.00, 0.01, 0.05, 0.10",
        "- Sigma 0.00 is sourced from frozen clean evidence.",
        "- New noisy episodes: 1080",
        "- Conceptual total including clean references: 1440 episodes.",
        "- PPO receives noisy observation; safety layer receives true simulator state.",
        "",
        "## Aggregate robustness",
        "",
        "| Domain | Method | Sigma | Violation rate | Delta vs clean | Reward | Reward delta | Success | Intervention |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in aggregates:
        lines.append(
            "| "
            f"{row['domain']} | "
            f"{row['method']} | "
            f"{row['sigma']:.2f} | "
            f"{row['violation_step_rate']['mean']:.6f} | "
            f"{row['violation_delta_from_clean']['mean']:+.6f} | "
            f"{row['reward']['mean']:.6f} | "
            f"{row['reward_delta_from_clean']['mean']:+.6f} | "
            f"{row['success_rate']['mean']:.6f} | "
            f"{row['intervention_rate']['mean']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Mechanism findings",
            "",
            (
                "- Noisy LYAPUNOV strict-decrease steps: "
                f"{findings['noisy_lyapunov_strict_decrease_steps']}"
            ),
            (
                "- Noisy LYAPUNOV reason counts: "
                f"`{findings['noisy_lyapunov_intervention_reason_counts']}`"
            ),
            (
                "- Robotics grasped-state steps under noisy LYAPUNOV runs: "
                f"{findings['noisy_lyapunov_grasped_steps']}"
            ),
            "",
            "## Limitations",
            "",
            "- Gaussian noise is an uncalibrated synthetic state-coordinate perturbation.",
            "- It is not a production sensor-error model.",
            "- The policy receives noisy observations while the safety layer retains true simulator state.",
            "- No formal robustness, certified safety, or real-sensor robustness claim is made.",
            "",
        ]
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    clean_cells = _load_clean_cells()

    (
        noisy_cells,
        noisy_reason_counts,
        strict_decrease_steps,
        grasped_steps,
        lyapunov_total_steps,
    ) = _load_noisy_cells()

    cells = clean_cells + noisy_cells

    if len(cells) != 72:
        raise RuntimeError(f"expected 72 conceptual cells, " f"found {len(cells)}")

    _add_clean_deltas(cells)

    aggregates = _aggregate_cells(cells)

    if len(aggregates) != 24:
        raise RuntimeError(f"expected 24 aggregate rows, " f"found {len(aggregates)}")

    findings = {
        "noisy_lyapunov_intervention_reason_counts": dict(
            sorted(noisy_reason_counts.items())
        ),
        "noisy_lyapunov_strict_decrease_steps": (strict_decrease_steps),
        "noisy_lyapunov_grasped_steps": (grasped_steps),
        "noisy_lyapunov_environment_steps": (lyapunov_total_steps),
        "principal_lyapunov_decrease_interventions_observed": (
            noisy_reason_counts.get(
                "lyapunov_decrease",
                0,
            )
            + noisy_reason_counts.get(
                "LYAPUNOV_DECREASE",
                0,
            )
            > 0
        ),
    }

    payload = {
        "artifact": ("sprint5-gaussian-robustness-summary"),
        "sprint": "5.10",
        "condition": ("gaussian_state_perturbation"),
        "principal_seeds": list(SEEDS),
        "evaluation_seeds": list(
            range(
                20_000,
                20_020,
            )
        ),
        "sigmas": list(SIGMAS),
        "new_noisy_cells": 54,
        "new_noisy_episodes": 1080,
        "conceptual_cells": 72,
        "conceptual_episodes": 1440,
        "sigma_zero_from_frozen_clean_evidence": True,
        "policy_uses_noisy_observation": True,
        "safety_layer_uses_true_state": True,
        "cells": cells,
        "aggregates": aggregates,
        "findings": findings,
        "claims": {
            "formal_robustness_guarantee": False,
            "production_sensor_noise_model": False,
            "real_sensor_robustness_claim": False,
            "certified_safety": False,
            "production_safety_claim": False,
            "quantum_robustness_advantage_claim": False,
        },
    }

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = OUTPUT_ROOT / "sprint5-gaussian-robustness-summary.json"

    csv_path = OUTPUT_ROOT / "sprint5-gaussian-robustness-summary.csv"

    md_path = OUTPUT_ROOT / "sprint5-gaussian-robustness-summary.md"

    json_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    _write_csv(
        cells,
        csv_path,
    )

    _write_markdown(
        aggregates,
        findings,
        md_path,
    )

    print("Clean seed cells: " f"{len(clean_cells)}")

    print("Noisy seed cells: " f"{len(noisy_cells)}")

    print("Conceptual seed cells: " f"{len(cells)}")

    print("Aggregate rows: " f"{len(aggregates)}")

    print()
    print(
        "Noisy LYAPUNOV reason counts:",
        findings["noisy_lyapunov_intervention_reason_counts"],
    )

    print(
        "Strict Lyapunov-decrease steps:",
        strict_decrease_steps,
    )

    print(
        "Noisy robotics grasped steps:",
        grasped_steps,
    )

    print()
    print("SPRINT 5.10 GAUSSIAN " "ROBUSTNESS SUMMARY: PASS")


if __name__ == "__main__":
    main()
