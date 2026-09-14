"""Evidence utilities for the Sprint 5.4 no-filter safety baseline."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from q_vla_forge.rl.ppo import GaussianActorCritic
from q_vla_forge.safety.contracts import (
    RobustnessCondition,
    SafetyMethod,
)

PRINCIPAL_SEEDS = (
    42,
    123,
    456,
)

SAFETY_EVALUATION_SEEDS = tuple(
    range(
        20_000,
        20_020,
    )
)

EXPECTED_TRAINING_BUDGET = 20_000

EXPECTED_CHECKPOINT_SELECTION = "final_20000_step_policy"


@dataclass(frozen=True)
class LoadedPPOPolicy:
    """One validated frozen PPO policy used by Sprint 5 safety evaluation."""

    domain: str
    principal_seed: int

    checkpoint_path: Path
    checkpoint_sha256: str

    recovery_record_path: Path
    recovery_record_sha256: str

    model: GaussianActorCritic

    action_low: np.ndarray
    action_high: np.ndarray

    training_budget: int
    checkpoint_selection: str


@dataclass(frozen=True)
class SafetyEpisodeEvidence:
    """Episode-level raw evidence for one Sprint 5 safety evaluation."""

    domain: str
    principal_seed: int
    evaluation_seed: int

    method: SafetyMethod
    robustness_condition: RobustnessCondition

    reward: float
    success: bool
    episode_length: int

    violation_step_count: int
    constraint_violation_count: int
    critical_violation_step_count: int

    intervention_count: int

    total_action_correction_l2: float
    max_action_correction_l2: float

    policy_action_out_of_bounds_count: int

    category_violation_counts: dict[
        str,
        int,
    ]


@dataclass(frozen=True)
class SafetySeedSummary:
    """Aggregated evidence for one domain and one principal PPO seed."""

    domain: str
    principal_seed: int

    method: SafetyMethod
    robustness_condition: RobustnessCondition

    episode_count: int
    total_environment_steps: int

    mean_reward: float
    reward_sample_sd: float

    success_rate: float
    mean_episode_length: float

    violation_step_count: int
    violation_step_rate: float

    constraint_violation_count: int
    constraint_violation_rate: float

    critical_violation_step_count: int
    critical_violation_step_rate: float

    intervention_count: int
    intervention_rate: float

    mean_action_correction_l2: float
    p95_action_correction_l2: float
    max_action_correction_l2: float

    policy_action_out_of_bounds_count: int

    category_violation_counts: dict[
        str,
        int,
    ]

    category_violation_rates: dict[
        str,
        float,
    ]


def sha256_file(
    path: Path,
) -> str:
    """Return SHA256 for one evidence file."""

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(65_536),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _require_finite(
    value: float,
    *,
    name: str,
) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


def _sample_sd(
    values: list[float],
) -> float:
    if not values:
        raise ValueError("sample SD requires at least one value")

    if len(values) == 1:
        return 0.0

    return float(statistics.stdev(values))


def _p95(
    values: list[float],
) -> float:
    if not values:
        return 0.0

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    return float(
        np.percentile(
            array,
            95,
        )
    )


def action_is_within_bounds(
    *,
    action: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    atol: float = 1e-7,
) -> bool:
    """Return whether an action lies inside the frozen environment bounds."""

    action_array = np.asarray(
        action,
        dtype=np.float64,
    )

    low = np.asarray(
        action_low,
        dtype=np.float64,
    )

    high = np.asarray(
        action_high,
        dtype=np.float64,
    )

    if action_array.shape != low.shape or action_array.shape != high.shape:
        raise ValueError("action and bounds must have matching shapes")

    if not np.all(np.isfinite(action_array)):
        raise ValueError("action must contain only finite values")

    if not np.all(np.isfinite(low)):
        raise ValueError("lower bounds must be finite")

    if not np.all(np.isfinite(high)):
        raise ValueError("upper bounds must be finite")

    if np.any(high <= low):
        raise ValueError("each action upper bound must exceed its lower bound")

    return bool(
        np.all(action_array >= low - atol) and np.all(action_array <= high + atol)
    )


def load_frozen_ppo_policy(
    *,
    root: Path,
    domain: str,
    principal_seed: int,
) -> LoadedPPOPolicy:
    """Load one validated reconstructed Sprint 4 PPO policy."""

    if principal_seed not in PRINCIPAL_SEEDS:
        raise ValueError(f"unsupported principal seed: {principal_seed}")

    if domain not in {
        "autonomous_driving",
        "robotics",
    }:
        raise ValueError(f"unsupported domain: {domain}")

    checkpoint_path = (
        root
        / "results"
        / "rl"
        / "checkpoints"
        / f"{domain}-seed-{principal_seed}-final.pt"
    )

    recovery_record_path = (
        root
        / "results"
        / "rl"
        / "checkpoint-recovery"
        / f"{domain}-seed-{principal_seed}-recovery.json"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"PPO checkpoint missing: {checkpoint_path}")

    if not recovery_record_path.exists():
        raise FileNotFoundError(
            "checkpoint recovery record missing: " f"{recovery_record_path}"
        )

    recovery = json.loads(recovery_record_path.read_text(encoding="utf-8"))

    if recovery["domain"] != domain:
        raise ValueError("recovery domain does not match requested domain")

    if recovery["seed"] != principal_seed:
        raise ValueError("recovery seed does not match requested seed")

    if recovery["training_budget"] != EXPECTED_TRAINING_BUDGET:
        raise ValueError("recovery training budget is not 20,000")

    if recovery["checkpoint_selection"] != EXPECTED_CHECKPOINT_SELECTION:
        raise ValueError("recovery checkpoint is not the frozen final policy")

    if recovery["historical_trajectory_match"] is not True:
        raise ValueError(
            "checkpoint recovery did not reproduce " "the historical PPO trajectory"
        )

    if recovery["maximum_absolute_metric_delta"] != 0.0:
        raise ValueError("checkpoint recovery historical metric delta is non-zero")

    checkpoint_hash = sha256_file(checkpoint_path)

    if checkpoint_hash != recovery["checkpoint_sha256"]:
        raise ValueError("checkpoint SHA256 does not match recovery record")

    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if payload["policy_family"] != "classical_ppo":
        raise ValueError("checkpoint is not a classical PPO policy")

    if payload["domain"] != domain:
        raise ValueError("checkpoint domain mismatch")

    if payload["seed"] != principal_seed:
        raise ValueError("checkpoint seed mismatch")

    if payload["total_environment_steps"] != EXPECTED_TRAINING_BUDGET:
        raise ValueError("checkpoint does not represent final 20k policy")

    if payload["checkpoint_selection"] != EXPECTED_CHECKPOINT_SELECTION:
        raise ValueError("checkpoint selection metadata mismatch")

    action_low = np.asarray(
        payload["action_low"],
        dtype=np.float32,
    )

    action_high = np.asarray(
        payload["action_high"],
        dtype=np.float32,
    )

    if action_low.shape != action_high.shape:
        raise ValueError("checkpoint action bounds have inconsistent shapes")

    model = GaussianActorCritic(
        observation_dim=int(payload["observation_dim"]),
        action_low=action_low,
        action_high=action_high,
        actor_hidden_dim=int(payload["actor_hidden_dim"]),
        critic_hidden_dim=int(payload["critic_hidden_dim"]),
        initial_log_std=float(payload["initial_log_std"]),
    )

    model.load_state_dict(payload["model_state_dict"])

    model.eval()

    return LoadedPPOPolicy(
        domain=domain,
        principal_seed=principal_seed,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_hash,
        recovery_record_path=(recovery_record_path),
        recovery_record_sha256=(sha256_file(recovery_record_path)),
        model=model,
        action_low=action_low.copy(),
        action_high=action_high.copy(),
        training_budget=(EXPECTED_TRAINING_BUDGET),
        checkpoint_selection=(EXPECTED_CHECKPOINT_SELECTION),
    )


def deterministic_policy_action(
    *,
    policy: LoadedPPOPolicy,
    observation: np.ndarray,
) -> np.ndarray:
    """Compute the deterministic PPO actor-mean action."""

    observation_array = np.asarray(
        observation,
        dtype=np.float32,
    )

    if observation_array.ndim != 1:
        raise ValueError("observation must be one-dimensional")

    if not np.all(np.isfinite(observation_array)):
        raise ValueError("observation must contain only finite values")

    with torch.no_grad():
        observation_tensor = torch.as_tensor(
            observation_array,
            dtype=torch.float32,
        ).unsqueeze(0)

        action = policy.model.deterministic_action(observation_tensor)[0].cpu().numpy()

    return np.asarray(
        action,
        dtype=np.float64,
    )


def summarize_episode_evidence(
    episodes: list[SafetyEpisodeEvidence],
) -> SafetySeedSummary:
    """Aggregate episode-level evidence for one principal safety cell."""

    if not episodes:
        raise ValueError("at least one episode is required")

    first = episodes[0]

    if first.method != SafetyMethod.NONE:
        raise ValueError("Sprint 5.4 summary requires SafetyMethod.NONE")

    if first.robustness_condition != RobustnessCondition.CLEAN:
        raise ValueError("Sprint 5.4 summary requires CLEAN condition")

    expected_domain = first.domain

    expected_seed = first.principal_seed

    evaluation_seeds: set[int] = set()

    total_steps = 0
    rewards: list[float] = []
    successes: list[float] = []
    lengths: list[float] = []

    violation_step_count = 0
    constraint_violation_count = 0
    critical_violation_step_count = 0

    intervention_count = 0

    action_corrections: list[float] = []

    policy_action_out_of_bounds_count = 0

    category_counts: dict[
        str,
        int,
    ] = {}

    for episode in episodes:
        if episode.domain != expected_domain:
            raise ValueError("episodes contain multiple domains")

        if episode.principal_seed != expected_seed:
            raise ValueError("episodes contain multiple principal seeds")

        if episode.method != SafetyMethod.NONE:
            raise ValueError("episodes contain a non-NONE safety method")

        if episode.robustness_condition != RobustnessCondition.CLEAN:
            raise ValueError("episodes contain a non-clean robustness condition")

        if episode.evaluation_seed in evaluation_seeds:
            raise ValueError("duplicate evaluation seed")

        evaluation_seeds.add(episode.evaluation_seed)

        if episode.episode_length <= 0:
            raise ValueError("episode length must be positive")

        for name, value in (
            (
                "reward",
                episode.reward,
            ),
            (
                "total action correction",
                episode.total_action_correction_l2,
            ),
            (
                "maximum action correction",
                episode.max_action_correction_l2,
            ),
        ):
            _require_finite(
                value,
                name=name,
            )

        if episode.violation_step_count < 0:
            raise ValueError("violation step count cannot be negative")

        if episode.constraint_violation_count < 0:
            raise ValueError("constraint violation count cannot be negative")

        if episode.critical_violation_step_count < 0:
            raise ValueError("critical violation step count cannot be negative")

        if episode.violation_step_count > episode.episode_length:
            raise ValueError("violation steps cannot exceed episode length")

        if episode.critical_violation_step_count > episode.violation_step_count:
            raise ValueError("critical violation steps cannot exceed violation steps")

        if episode.intervention_count != 0:
            raise ValueError("NONE baseline cannot contain interventions")

        if (
            episode.total_action_correction_l2 != 0.0
            or episode.max_action_correction_l2 != 0.0
        ):
            raise ValueError("NONE baseline cannot contain action corrections")

        total_steps += episode.episode_length

        rewards.append(float(episode.reward))

        successes.append(float(episode.success))

        lengths.append(float(episode.episode_length))

        violation_step_count += episode.violation_step_count

        constraint_violation_count += episode.constraint_violation_count

        critical_violation_step_count += episode.critical_violation_step_count

        intervention_count += episode.intervention_count

        action_corrections.extend([0.0] * episode.episode_length)

        policy_action_out_of_bounds_count += episode.policy_action_out_of_bounds_count

        for (
            category,
            count,
        ) in episode.category_violation_counts.items():
            if count < 0:
                raise ValueError("category violation count cannot be negative")

            category_counts[category] = (
                category_counts.get(
                    category,
                    0,
                )
                + count
            )

    if total_steps <= 0:
        raise ValueError("total environment steps must be positive")

    category_rates = {
        category: (count / total_steps)
        for (
            category,
            count,
        ) in sorted(category_counts.items())
    }

    return SafetySeedSummary(
        domain=expected_domain,
        principal_seed=(expected_seed),
        method=SafetyMethod.NONE,
        robustness_condition=(RobustnessCondition.CLEAN),
        episode_count=len(episodes),
        total_environment_steps=(total_steps),
        mean_reward=float(statistics.mean(rewards)),
        reward_sample_sd=(_sample_sd(rewards)),
        success_rate=float(statistics.mean(successes)),
        mean_episode_length=float(statistics.mean(lengths)),
        violation_step_count=(violation_step_count),
        violation_step_rate=(violation_step_count / total_steps),
        constraint_violation_count=(constraint_violation_count),
        constraint_violation_rate=(constraint_violation_count / total_steps),
        critical_violation_step_count=(critical_violation_step_count),
        critical_violation_step_rate=(critical_violation_step_count / total_steps),
        intervention_count=(intervention_count),
        intervention_rate=(intervention_count / total_steps),
        mean_action_correction_l2=float(statistics.mean(action_corrections)),
        p95_action_correction_l2=(_p95(action_corrections)),
        max_action_correction_l2=float(max(action_corrections)),
        policy_action_out_of_bounds_count=(policy_action_out_of_bounds_count),
        category_violation_counts=(dict(sorted(category_counts.items()))),
        category_violation_rates=(category_rates),
    )


def episode_evidence_to_dict(
    episode: SafetyEpisodeEvidence,
) -> dict[str, Any]:
    """Convert one episode record into JSON-safe evidence."""

    payload = asdict(episode)

    payload["method"] = episode.method.value

    payload["robustness_condition"] = episode.robustness_condition.value

    return payload


def seed_summary_to_dict(
    summary: SafetySeedSummary,
) -> dict[str, Any]:
    """Convert one seed summary into JSON-safe evidence."""

    payload = asdict(summary)

    payload["method"] = summary.method.value

    payload["robustness_condition"] = summary.robustness_condition.value

    return payload


from q_vla_forge.rl.driving_env import DrivingRLEnv
from q_vla_forge.rl.robotics_env import RoboticsRLEnv
from q_vla_forge.safety.contracts import (
    ViolationRecord,
    ViolationSeverity,
)
from q_vla_forge.safety.driving_constraints import (
    evaluate_driving_violations,
)
from q_vla_forge.safety.no_filter import (
    no_filter_decision,
)
from q_vla_forge.safety.robotics_constraints import (
    evaluate_robotics_violations,
)


def evaluate_domain_violations(
    *,
    domain: str,
    state: np.ndarray,
    action: np.ndarray,
) -> tuple[ViolationRecord, ...]:
    """Route one state-action pair through the frozen domain safety contract."""

    if domain == "autonomous_driving":
        return evaluate_driving_violations(
            state,
            action,
        )

    if domain == "robotics":
        return evaluate_robotics_violations(
            state,
            action,
        )

    raise ValueError(f"unsupported domain: {domain}")


def environment_for_domain(
    domain: str,
):
    """Create the frozen Sprint 4 RL environment for one domain."""

    if domain == "autonomous_driving":
        return DrivingRLEnv()

    if domain == "robotics":
        return RoboticsRLEnv()

    raise ValueError(f"unsupported domain: {domain}")


def run_no_filter_episode(
    *,
    policy: LoadedPPOPolicy,
    evaluation_seed: int,
) -> SafetyEpisodeEvidence:
    """Run one deterministic clean-condition NONE safety episode."""

    if evaluation_seed not in SAFETY_EVALUATION_SEEDS:
        raise ValueError("evaluation seed is outside the frozen Sprint 5.1 set")

    env = environment_for_domain(policy.domain)

    observation, _ = env.reset(seed=evaluation_seed)

    total_reward = 0.0
    episode_length = 0

    violation_step_count = 0
    constraint_violation_count = 0
    critical_violation_step_count = 0

    intervention_count = 0

    total_action_correction_l2 = 0.0
    max_action_correction_l2 = 0.0

    policy_action_out_of_bounds_count = 0

    category_violation_counts: dict[
        str,
        int,
    ] = {}

    final_success = False

    while True:
        true_state = np.asarray(
            observation,
            dtype=np.float64,
        ).copy()

        observed_state = true_state.copy()

        proposed_action = deterministic_policy_action(
            policy=policy,
            observation=observed_state,
        )

        if not action_is_within_bounds(
            action=proposed_action,
            action_low=policy.action_low,
            action_high=policy.action_high,
        ):
            policy_action_out_of_bounds_count += 1

        violations = evaluate_domain_violations(
            domain=policy.domain,
            state=true_state,
            action=proposed_action,
        )

        decision = no_filter_decision(
            proposed_action=(proposed_action),
            violations=violations,
        )

        violated_records = tuple(record for record in violations if record.violated)

        if violated_records:
            violation_step_count += 1

        constraint_violation_count += len(violated_records)

        if any(
            record.severity == ViolationSeverity.CRITICAL for record in violated_records
        ):
            critical_violation_step_count += 1

        for record in violated_records:
            category_violation_counts[record.name] = (
                category_violation_counts.get(
                    record.name,
                    0,
                )
                + 1
            )

        if decision.intervened:
            intervention_count += 1

        total_action_correction_l2 += decision.correction_l2

        max_action_correction_l2 = max(
            max_action_correction_l2,
            decision.correction_l2,
        )

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(decision.executed_action)

        total_reward += float(reward)

        episode_length += 1

        if terminated or truncated:
            final_success = bool(info["success"])
            break

    env.close()

    return SafetyEpisodeEvidence(
        domain=policy.domain,
        principal_seed=(policy.principal_seed),
        evaluation_seed=(evaluation_seed),
        method=SafetyMethod.NONE,
        robustness_condition=(RobustnessCondition.CLEAN),
        reward=total_reward,
        success=final_success,
        episode_length=(episode_length),
        violation_step_count=(violation_step_count),
        constraint_violation_count=(constraint_violation_count),
        critical_violation_step_count=(critical_violation_step_count),
        intervention_count=(intervention_count),
        total_action_correction_l2=(total_action_correction_l2),
        max_action_correction_l2=(max_action_correction_l2),
        policy_action_out_of_bounds_count=(policy_action_out_of_bounds_count),
        category_violation_counts=(dict(sorted(category_violation_counts.items()))),
    )
