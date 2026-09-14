"""Verify Sprint 5.10 Gaussian robustness readiness."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from q_vla_forge.evaluation.safety_baseline import (
    load_frozen_ppo_policy,
)
from q_vla_forge.safety.perturbations import (
    GAUSSIAN_NOISE_LEVELS,
    GAUSSIAN_PRINCIPAL_NOISY_LEVELS,
    apply_gaussian_observation_noise,
    make_gaussian_rng,
)

ROOT = Path(__file__).resolve().parents[1]

DOMAINS = (
    "autonomous_driving",
    "robotics",
)

PRINCIPAL_SEEDS = (
    42,
    123,
    456,
)

EXPECTED_LEVELS = (
    0.0,
    0.01,
    0.05,
    0.10,
)

EXPECTED_PRINCIPAL_LEVELS = (
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


def _check_checkpoints() -> None:
    for domain in DOMAINS:
        for seed in PRINCIPAL_SEEDS:
            policy = load_frozen_ppo_policy(
                root=ROOT,
                domain=domain,
                principal_seed=seed,
            )

            _require(
                policy.checkpoint_path.exists(),
                (f"checkpoint exists " f"{domain} seed {seed}"),
            )

            _require(
                bool(policy.checkpoint_sha256),
                (f"checkpoint SHA present " f"{domain} seed {seed}"),
            )

            _require(
                policy.training_budget == 20_000,
                (f"training budget frozen " f"{domain} seed {seed}"),
            )


def _check_clean_evidence() -> None:
    paths = ROOT / "results" / "safety"

    expected = (
        paths / "baseline" / "sprint5-no-filter-safety-summary.json",
        paths / "clipping" / "sprint5-clipping-safety-summary.json",
        paths / "lyapunov-driving" / "sprint5-driving-lyapunov-summary.json",
        paths / "lyapunov-robotics" / "sprint5-robotics-lyapunov-summary.json",
    )

    for path in expected:
        _require(
            path.exists(),
            ("clean evidence exists: " f"{path.relative_to(ROOT)}"),
        )


def _check_noise_determinism() -> None:
    for domain, dimension in (
        (
            "autonomous_driving",
            4,
        ),
        (
            "robotics",
            6,
        ),
    ):
        state = np.linspace(
            -0.5,
            0.5,
            dimension,
            dtype=np.float32,
        )

        for sigma in (
            0.0,
            0.01,
            0.05,
            0.10,
        ):
            first_rng, first_seed = make_gaussian_rng(
                domain=domain,
                principal_seed=42,
                evaluation_seed=20_000,
                sigma=sigma,
            )

            second_rng, second_seed = make_gaussian_rng(
                domain=domain,
                principal_seed=42,
                evaluation_seed=20_000,
                sigma=sigma,
            )

            first_observed, first_noise = apply_gaussian_observation_noise(
                true_state=state,
                sigma=sigma,
                rng=first_rng,
            )

            second_observed, second_noise = apply_gaussian_observation_noise(
                true_state=state,
                sigma=sigma,
                rng=second_rng,
            )

            _require(
                first_seed == second_seed,
                (f"deterministic noise seed " f"{domain} sigma {sigma:.2f}"),
            )

            _require(
                np.array_equal(
                    first_noise,
                    second_noise,
                ),
                (f"deterministic noise vector " f"{domain} sigma {sigma:.2f}"),
            )

            _require(
                np.array_equal(
                    first_observed,
                    second_observed,
                ),
                (f"deterministic observation " f"{domain} sigma {sigma:.2f}"),
            )

            _require(
                np.array_equal(
                    first_observed,
                    state + first_noise,
                ),
                (f"observed=true+noise " f"{domain} sigma {sigma:.2f}"),
            )

            if sigma == 0.0:
                _require(
                    np.array_equal(
                        first_observed,
                        state,
                    ),
                    (f"sigma-zero identity " f"{domain}"),
                )


def _check_runner_source() -> None:
    path = ROOT / "experiments" / "run_gaussian_robustness.py"

    _require(
        path.exists(),
        "Gaussian runner exists",
    )

    source = path.read_text(encoding="utf-8")

    forbidden = (
        "optimizer.",
        ".backward(",
        "train_ppo(",
        "run_ppo_training(",
    )

    for token in forbidden:
        _require(
            token not in source,
            ("runner excludes " f"{token}"),
        )

    required = (
        "deterministic_policy_action",
        "apply_clipping_safety_filter",
        "apply_lyapunov_safety_filter",
        "apply_gaussian_observation_noise",
        "make_gaussian_rng",
        "true_state=true_state",
        "observation=observed_state",
        "object_grasped",
    )

    for token in required:
        _require(
            token in source,
            ("runner contains " f"{token}"),
        )


def main() -> None:
    print("=" * 64)
    print(" Q-VLA FORGE - " "SPRINT 5.10 GAUSSIAN ROBUSTNESS READINESS")
    print("=" * 64)
    print()

    _require(
        GAUSSIAN_NOISE_LEVELS == EXPECTED_LEVELS,
        "Gaussian levels frozen",
    )

    _require(
        GAUSSIAN_PRINCIPAL_NOISY_LEVELS == EXPECTED_PRINCIPAL_LEVELS,
        "principal noisy levels frozen",
    )

    _require(
        len(DOMAINS) == 2,
        "two domains locked",
    )

    _require(
        PRINCIPAL_SEEDS
        == (
            42,
            123,
            456,
        ),
        "principal seeds frozen",
    )

    _check_checkpoints()
    _check_clean_evidence()
    _check_noise_determinism()
    _check_runner_source()

    print()
    print("Safety protocol: PASS")
    print("Frozen checkpoints: PASS")
    print("Clean evidence: PASS")
    print("Gaussian perturbation: PASS")
    print("Paired deterministic noise: PASS")
    print("True-state safety access: PASS")
    print("Robotics grasp context: PASS")
    print("No training: PASS")
    print()
    print("SPRINT 5.10 " "GAUSSIAN ROBUSTNESS READINESS: PASS")


if __name__ == "__main__":
    main()
