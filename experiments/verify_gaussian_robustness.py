"""Independently verify Sprint 5.10 Gaussian robustness evidence."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

RUNS = ROOT / "results" / "safety" / "gaussian-robustness" / "runs"

SUMMARY = (
    ROOT
    / "results"
    / "safety"
    / "gaussian-robustness"
    / "sprint5-gaussian-robustness-summary.json"
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

SEEDS = (
    42,
    123,
    456,
)

SIGMAS = (
    0.01,
    0.05,
    0.10,
)


def _require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise AssertionError(message)

    print(f"[PASS] {message}")


def _close(
    left: float,
    right: float,
    *,
    message: str,
) -> None:
    _require(
        math.isclose(
            left,
            right,
            rel_tol=0.0,
            abs_tol=1.0e-10,
        ),
        message,
    )


def _sample_sd(
    values: list[float],
) -> float:
    if len(values) <= 1:
        return 0.0

    return float(stdev(values))


def _slug(
    domain: str,
) -> str:
    return domain.replace(
        "_",
        "-",
    )


def _path(
    *,
    domain: str,
    method: str,
    seed: int,
    sigma: float,
) -> Path:
    sigma_slug = f"{sigma:.2f}".replace(
        ".",
        "p",
    )

    return RUNS / (
        f"{_slug(domain)}-" f"{method}-" f"seed-{seed}-" f"sigma-{sigma_slug}.json"
    )


def _episode_arithmetic(
    episode: dict[str, Any],
) -> None:
    length = int(episode["episode_length"])

    _require(
        length > 0,
        "episode length positive",
    )

    if (
        int(episode["steps_object_grasped"]) + int(episode["steps_object_not_grasped"])
        > 0
    ):
        _require(
            int(episode["steps_object_grasped"])
            + int(episode["steps_object_not_grasped"])
            == length,
            "robotics grasp accounting",
        )


def main() -> None:
    print("=" * 72)
    print(" Q-VLA FORGE - SPRINT 5.10 " "GAUSSIAN ROBUSTNESS VERIFICATION")
    print("=" * 72)
    print()

    files = sorted(RUNS.glob("*.json"))

    _require(
        len(files) == 54,
        "54 / 54 noisy principal files",
    )

    expected_cells = {
        (
            domain,
            method,
            seed,
            sigma,
        )
        for domain in DOMAINS
        for method in METHODS
        for seed in SEEDS
        for sigma in SIGMAS
    }

    observed_cells = set()

    checkpoint_lookup: dict[
        tuple[str, int],
        str,
    ] = {}

    total_episodes = 0

    for domain in DOMAINS:
        for seed in SEEDS:
            for sigma in SIGMAS:
                paired_payloads = {}

                for method in METHODS:
                    path = _path(
                        domain=domain,
                        method=method,
                        seed=seed,
                        sigma=sigma,
                    )

                    _require(
                        path.exists(),
                        (f"cell exists " f"{domain} {method} " f"{seed} {sigma:.2f}"),
                    )

                    payload = json.loads(path.read_text(encoding="utf-8"))

                    observed_cells.add(
                        (
                            str(payload["domain"]),
                            str(payload["method"]),
                            int(payload["principal_seed"]),
                            float(payload["sigma"]),
                        )
                    )

                    _require(
                        payload["artifact_type"] == "principal",
                        "principal artifact identity",
                    )

                    _require(
                        payload["new_training_performed"] is False,
                        "no training",
                    )

                    _require(
                        payload["policy_fine_tuning_performed"] is False,
                        "no fine-tuning",
                    )

                    _require(
                        payload["filter_tuning_performed"] is False,
                        "no filter tuning",
                    )

                    _require(
                        payload["policy_uses_observed_state"] is True,
                        "policy uses noisy observation",
                    )

                    _require(
                        payload["safety_uses_true_state"] is True,
                        "safety uses true state",
                    )

                    _require(
                        payload["observation_clipped_after_noise"] is False,
                        "no observation clipping",
                    )

                    episodes = payload["episodes"]

                    _require(
                        len(episodes) == 20,
                        "20 episodes per cell",
                    )

                    total_episodes += len(episodes)

                    sha = str(payload["checkpoint_sha256"])

                    checkpoint_key = (
                        domain,
                        seed,
                    )

                    previous = checkpoint_lookup.get(checkpoint_key)

                    if previous is None:
                        checkpoint_lookup[checkpoint_key] = sha
                    else:
                        _require(
                            previous == sha,
                            "checkpoint fairness",
                        )

                    for episode in episodes:
                        _episode_arithmetic(episode)

                    paired_payloads[method] = payload

                none_episodes = paired_payloads["none"]["episodes"]

                for method in (
                    "clipping",
                    "lyapunov",
                ):
                    method_episodes = paired_payloads[method]["episodes"]

                    for left, right in zip(
                        none_episodes,
                        method_episodes,
                        strict=True,
                    ):
                        _require(
                            left["noise_seed"] == right["noise_seed"],
                            "paired episode noise seed",
                        )

                        left_audits = left["audit_snapshots"]

                        right_audits = right["audit_snapshots"]

                        for (
                            left_step,
                            right_step,
                        ) in zip(
                            left_audits,
                            right_audits,
                            strict=True,
                        ):
                            _require(
                                left_step["noise_vector"] == right_step["noise_vector"],
                                ("paired cross-method " "audit noise"),
                            )

    _require(
        observed_cells == expected_cells,
        "complete 54-cell matrix",
    )

    _require(
        total_episodes == 1080,
        "1080 / 1080 noisy episodes",
    )

    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

    _require(
        summary["conceptual_cells"] == 72,
        "72 conceptual cells",
    )

    _require(
        summary["conceptual_episodes"] == 1440,
        "1440 conceptual episodes",
    )

    _require(
        summary["sigma_zero_from_frozen_clean_evidence"] is True,
        "sigma zero sourced from clean evidence",
    )

    cells = summary["cells"]

    _require(
        len(cells) == 72,
        "summary contains 72 seed cells",
    )

    clean_cells = [cell for cell in cells if float(cell["sigma"]) == 0.0]

    _require(
        len(clean_cells) == 18,
        "18 frozen clean seed cells",
    )

    noisy_cells = [cell for cell in cells if float(cell["sigma"]) > 0.0]

    _require(
        len(noisy_cells) == 54,
        "54 noisy summary cells",
    )

    aggregates = summary["aggregates"]

    _require(
        len(aggregates) == 24,
        "24 domain-method-sigma aggregates",
    )

    for row in aggregates:
        matching = [
            cell
            for cell in cells
            if (
                cell["domain"] == row["domain"]
                and cell["method"] == row["method"]
                and float(cell["sigma"]) == float(row["sigma"])
            )
        ]

        _require(
            len(matching) == 3,
            "aggregate contains three seeds",
        )

        values = [float(cell["violation_step_rate"]) for cell in matching]

        _close(
            float(row["violation_step_rate"]["mean"]),
            float(mean(values)),
            message=("aggregate violation mean"),
        )

        _close(
            float(row["violation_step_rate"]["sample_sd"]),
            _sample_sd(values),
            message=("aggregate violation sample SD"),
        )

    claims = summary["claims"]

    for key in (
        "formal_robustness_guarantee",
        "production_sensor_noise_model",
        "real_sensor_robustness_claim",
        "certified_safety",
        "production_safety_claim",
        "quantum_robustness_advantage_claim",
    ):
        _require(
            claims[key] is False,
            (f"claim controlled: {key}"),
        )

    findings = summary["findings"]

    print()
    print(
        "Noisy Lyapunov reason counts:",
        findings["noisy_lyapunov_intervention_reason_counts"],
    )

    print(
        "Noisy strict-decrease steps:",
        findings["noisy_lyapunov_strict_decrease_steps"],
    )

    print(
        "Noisy Lyapunov grasped steps:",
        findings["noisy_lyapunov_grasped_steps"],
    )

    print()
    print("Protocol: PASS")
    print("Matrix completeness: PASS")
    print("Checkpoint fairness: PASS")
    print("Cross-method noise pairing: PASS")
    print("True-state integrity: PASS")
    print("Clean-reference integration: PASS")
    print("Robustness accounting: PASS")
    print("Claim controls: PASS")
    print()
    print("SPRINT 5.10 GAUSSIAN " "ROBUSTNESS: PASS")


if __name__ == "__main__":
    main()
