"""Sprint 7.4 QML ablation contract and validation helpers."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REQUIRED_SEEDS: tuple[int, ...] = (
    42,
    123,
    456,
)

DOMAINS: tuple[str, ...] = (
    "autonomous_driving",
    "robotics",
)

POLICIES: tuple[str, ...] = (
    "ppo_mlp",
    "ppo_pqc",
)

PRIMARY_METRIC = "environment_steps_to_target"

SECONDARY_METRICS: tuple[str, ...] = (
    "episodes_to_target",
    "final_evaluation_reward",
    "final_success_rate",
    "training_seconds",
    "policy_parameters",
)

EXECUTION_BACKEND = "simulator"
QUANTUM_HARDWARE_USED = False

CLAIM_CONTROLS = {
    "quantum_advantage_claim_allowed": False,
    "quantum_speedup_claim_allowed": False,
    "qml_sample_efficiency_advantage_claim_allowed": False,
    "qml_performance_superiority_claim_allowed": False,
    "qml_convergence_superiority_claim_allowed": False,
    "qpu_advantage_claim_allowed": False,
    "production_vla_superiority_claim_allowed": False,
}


@dataclass(frozen=True)
class QMLAblationRecord:
    """One Sprint 7.4 QML ablation record."""

    domain: str
    policy: str
    seed: int
    target_reached: bool
    environment_steps_to_target: int | None
    episodes_to_target: int | None
    final_evaluation_reward: float | None
    final_success_rate: float | None
    training_seconds: float | None
    actor_parameters: int
    source_artifacts: tuple[str, ...]
    execution_backend: str = EXECUTION_BACKEND
    quantum_hardware_used: bool = QUANTUM_HARDWARE_USED

    def __post_init__(self) -> None:
        validate_domain(self.domain)
        validate_policy(self.policy)
        validate_seed(self.seed)

        validate_target_attainment(
            target_reached=self.target_reached,
            environment_steps_to_target=(self.environment_steps_to_target),
        )

        validate_optional_positive_int(
            self.episodes_to_target,
            name="episodes_to_target",
        )

        validate_optional_finite(
            self.final_evaluation_reward,
            name="final_evaluation_reward",
        )

        validate_optional_rate(
            self.final_success_rate,
            name="final_success_rate",
        )

        validate_optional_nonnegative(
            self.training_seconds,
            name="training_seconds",
        )

        validate_parameter_count(
            self.actor_parameters,
            name="actor_parameters",
        )

        validate_source_provenance(self.source_artifacts)

        if self.execution_backend != EXECUTION_BACKEND:
            raise ValueError("execution_backend must remain simulator.")

        if self.quantum_hardware_used is not False:
            raise ValueError("quantum_hardware_used must remain False.")

    def to_dict(self) -> dict[str, Any]:
        """Convert record to serializable form."""

        return asdict(self)


def validate_domain(
    domain: str,
) -> None:
    """Require one of the locked Phase 1 domains."""

    if domain not in DOMAINS:
        raise ValueError(f"Unsupported domain: {domain}")


def validate_policy(
    policy: str,
) -> None:
    """Require one of the locked comparison policies."""

    if policy not in POLICIES:
        raise ValueError(f"Unsupported policy: {policy}")


def validate_seed(
    seed: int,
) -> None:
    """Require one of the locked seeds."""

    if seed not in REQUIRED_SEEDS:
        raise ValueError(f"Unsupported seed: {seed}")


def validate_required_seeds(
    seeds: tuple[int, ...] | list[int],
) -> None:
    """Require exactly the locked three-seed set."""

    observed = tuple(sorted(seeds))

    if observed != REQUIRED_SEEDS:
        raise ValueError(f"Expected seeds {REQUIRED_SEEDS}, " f"observed {observed}")


def validate_target_attainment(
    *,
    target_reached: bool,
    environment_steps_to_target: int | None,
) -> None:
    """Enforce correct target non-attainment semantics."""

    if not isinstance(
        target_reached,
        bool,
    ):
        raise TypeError("target_reached must be boolean.")

    if target_reached:
        if environment_steps_to_target is None:
            raise ValueError("Reached target requires " "environment_steps_to_target.")

        if (
            not isinstance(
                environment_steps_to_target,
                int,
            )
            or isinstance(
                environment_steps_to_target,
                bool,
            )
            or environment_steps_to_target <= 0
        ):
            raise ValueError(
                "environment_steps_to_target must " "be a positive integer."
            )

        return

    if environment_steps_to_target is not None:
        raise ValueError(
            "Non-reaching policy must have " "environment_steps_to_target=None."
        )


def validate_parameter_count(
    value: int,
    *,
    name: str,
) -> None:
    """Require a positive deterministic parameter count."""

    if (
        not isinstance(
            value,
            int,
        )
        or isinstance(
            value,
            bool,
        )
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer.")


def validate_optional_positive_int(
    value: int | None,
    *,
    name: str,
) -> None:
    """Validate an optional positive integer."""

    if value is None:
        return

    if (
        not isinstance(
            value,
            int,
        )
        or isinstance(
            value,
            bool,
        )
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer.")


def validate_optional_finite(
    value: float | None,
    *,
    name: str,
) -> None:
    """Validate an optional finite numeric value."""

    if value is None:
        return

    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite.")


def validate_optional_nonnegative(
    value: float | None,
    *,
    name: str,
) -> None:
    """Validate an optional finite non-negative numeric value."""

    if value is None:
        return

    validate_optional_finite(
        value,
        name=name,
    )

    if float(value) < 0.0:
        raise ValueError(f"{name} must be non-negative.")


def validate_optional_rate(
    value: float | None,
    *,
    name: str,
) -> None:
    """Validate an optional rate in the inclusive [0, 1] range."""

    if value is None:
        return

    validate_optional_finite(
        value,
        name=name,
    )

    numeric = float(value)

    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be in [0, 1].")


def validate_source_provenance(
    source_artifacts: tuple[str, ...],
) -> None:
    """Require explicit frozen evidence provenance."""

    if not source_artifacts:
        raise ValueError("At least one source artifact is required.")

    if any(not source.strip() for source in source_artifacts):
        raise ValueError("Source artifact paths must be non-empty.")


def parameter_reduction_percent(
    *,
    classical_parameters: int,
    qml_parameters: int,
) -> float:
    """Calculate deterministic QML actor parameter reduction."""

    validate_parameter_count(
        classical_parameters,
        name="classical_parameters",
    )

    validate_parameter_count(
        qml_parameters,
        name="qml_parameters",
    )

    return 100.0 * (1.0 - (qml_parameters / classical_parameters))


def compactness_record(
    *,
    domain: str,
    classical_parameters: int,
    qml_parameters: int,
    source_artifacts: tuple[str, ...],
) -> dict[str, Any]:
    """Build deterministic actor compactness evidence."""

    validate_domain(domain)

    validate_parameter_count(
        classical_parameters,
        name="classical_parameters",
    )

    validate_parameter_count(
        qml_parameters,
        name="qml_parameters",
    )

    validate_source_provenance(source_artifacts)

    return {
        "domain": domain,
        "classical_actor_parameters": (classical_parameters),
        "qml_actor_parameters": qml_parameters,
        "parameter_reduction_percent": (
            parameter_reduction_percent(
                classical_parameters=(classical_parameters),
                qml_parameters=qml_parameters,
            )
        ),
        "statistical": False,
        "source_artifacts": source_artifacts,
    }


def claim_controls() -> dict[str, bool]:
    """Return the frozen Sprint 7.4 claim controls."""

    return dict(CLAIM_CONTROLS)


PPO_SUMMARY_SOURCE = "results/rl/ppo/sprint4-ppo-summary.json"
PPO_TARGETS_SOURCE = "results/rl/ppo/sprint4-ppo-targets.json"

QML_SUMMARY_SOURCES = {
    "autonomous_driving": ("results/rl/qml/sprint4-driving-qml-summary.json"),
    "robotics": ("results/rl/qml/sprint4-robotics-qml-summary.json"),
}

ABLATION_SOURCE = "results/rl/ablation/" "sprint4-classical-vs-qml-ablation.json"


@dataclass(frozen=True)
class MatchedClassicalControlRecord:
    """Parameter-matched classical Sprint 4 control."""

    domain: str
    seed: int
    target_reached: bool
    environment_steps_to_target: int | None
    episodes_to_target: int | None
    final_evaluation_reward: float
    final_success_rate: float
    training_seconds: float
    actor_parameters: int
    source_artifacts: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_domain(self.domain)

        validate_seed(self.seed)

        validate_target_attainment(
            target_reached=self.target_reached,
            environment_steps_to_target=(self.environment_steps_to_target),
        )

        validate_optional_positive_int(
            self.episodes_to_target,
            name="episodes_to_target",
        )

        validate_optional_finite(
            self.final_evaluation_reward,
            name="final_evaluation_reward",
        )

        validate_optional_rate(
            self.final_success_rate,
            name="final_success_rate",
        )

        validate_optional_nonnegative(
            self.training_seconds,
            name="training_seconds",
        )

        validate_parameter_count(
            self.actor_parameters,
            name="actor_parameters",
        )

        validate_source_provenance(self.source_artifacts)

    def to_dict(self) -> dict[str, Any]:
        """Convert control record to serializable form."""

        return asdict(self)


@dataclass(frozen=True)
class CanonicalQMLAblation:
    """Canonical Sprint 4 evidence used by Sprint 7.4."""

    primary_records: tuple[
        QMLAblationRecord,
        ...,
    ]

    matched_classical_controls: tuple[
        MatchedClassicalControlRecord,
        ...,
    ]

    compactness: tuple[
        dict[str, Any],
        ...,
    ]


def _load_json(
    path: Path,
) -> dict[str, Any]:
    """Load one canonical frozen JSON artifact."""

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


def _qml_run_path(
    *,
    domain: str,
    seed: int,
) -> str:
    """Return principal QML run path, excluding repro runs."""

    return f"results/rl/qml/" f"{domain}-seed-{seed}.json"


def _matched_run_path(
    *,
    domain: str,
    seed: int,
) -> str:
    """Return principal matched-classical run path."""

    return "results/rl/ablation/" f"matched-classical-{domain}-seed-{seed}.json"


def _find_seed_entry(
    entries: list[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    """Find exactly one frozen seed entry."""

    matches = [entry for entry in entries if int(entry["seed"]) == seed]

    if len(matches) != 1:
        raise ValueError(
            f"Expected one entry for seed {seed}, " f"found {len(matches)}."
        )

    return matches[0]


def _find_paired_record(
    entries: list[dict[str, Any]],
    *,
    domain: str,
    seed: int,
) -> dict[str, Any]:
    """Find one canonical matched/QML paired record."""

    matches = [
        entry
        for entry in entries
        if entry["domain"] == domain and int(entry["seed"]) == seed
    ]

    if len(matches) != 1:
        raise ValueError(f"Expected one paired record for " f"{domain}/seed-{seed}.")

    return matches[0]


def _first_target_reach(
    *,
    evaluations: list[dict[str, Any]],
    target_reward: float,
) -> tuple[
    bool,
    int | None,
    int | None,
]:
    """Find first evaluation meeting the frozen reward target."""

    for evaluation in evaluations:
        reward = float(evaluation["mean_reward"])

        if reward >= target_reward:
            steps = int(evaluation["environment_steps"])

            episodes = int(evaluation["completed_training_episodes"])

            if steps <= 0:
                raise ValueError(
                    "Target reach must occur " "after positive environment steps."
                )

            if episodes <= 0:
                raise ValueError(
                    "Target reach must occur " "after positive training episodes."
                )

            return (
                True,
                steps,
                episodes,
            )

    return (
        False,
        None,
        None,
    )


def _build_ppo_record(
    *,
    domain: str,
    seed: int,
    ppo_summary: dict[str, Any],
    ppo_targets: dict[str, Any],
) -> QMLAblationRecord:
    """Build one PPO/MLP record from frozen Sprint 4 summaries."""

    domain_summary = ppo_summary["domains"][domain]

    run = _find_seed_entry(
        domain_summary["runs"],
        seed=seed,
    )

    target_entry = _find_seed_entry(
        ppo_targets["domains"][domain]["seeds"],
        seed=seed,
    )

    steps = int(target_entry["ppo_environment_steps_to_target"])

    episodes = int(target_entry["ppo_episodes_to_target"])

    if steps != int(run["environment_steps_to_target"]):
        raise ValueError(f"{domain}/seed-{seed}: " "PPO steps-to-target mismatch.")

    if episodes != int(run["episodes_to_target"]):
        raise ValueError(f"{domain}/seed-{seed}: " "PPO episodes-to-target mismatch.")

    return QMLAblationRecord(
        domain=domain,
        policy="ppo_mlp",
        seed=seed,
        target_reached=True,
        environment_steps_to_target=steps,
        episodes_to_target=episodes,
        final_evaluation_reward=float(run["final_evaluation_reward"]),
        final_success_rate=float(run["final_success_rate"]),
        training_seconds=float(run["training_seconds"]),
        actor_parameters=int(run["actor_parameters"]),
        source_artifacts=(
            PPO_SUMMARY_SOURCE,
            PPO_TARGETS_SOURCE,
        ),
    )


def _build_qml_record(
    *,
    root: Path,
    domain: str,
    seed: int,
    ppo_targets: dict[str, Any],
    ablation: dict[str, Any],
) -> QMLAblationRecord:
    """Build one PQC/QML record from principal Sprint 4 run."""

    relative_path = _qml_run_path(
        domain=domain,
        seed=seed,
    )

    payload = _load_json(root / relative_path)

    if payload["domain"] != domain:
        raise ValueError("QML domain mismatch.")

    if int(payload["seed"]) != seed:
        raise ValueError("QML seed mismatch.")

    comparison = payload["comparison"]

    target_entry = _find_seed_entry(
        ppo_targets["domains"][domain]["seeds"],
        seed=seed,
    )

    frozen_target = float(target_entry["target_evaluation_reward"])

    if not math.isclose(
        float(comparison["target_evaluation_reward"]),
        frozen_target,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"{domain}/seed-{seed}: " "QML target differs from frozen PPO target."
        )

    qml_steps = comparison["qml_environment_steps_to_target"]

    qml_episodes = comparison["qml_episodes_to_target"]

    target_reached = qml_steps is not None

    paired = _find_paired_record(
        ablation["paired_records"],
        domain=domain,
        seed=seed,
    )

    if bool(paired["qml_target_reached"]) != target_reached:
        raise ValueError(f"{domain}/seed-{seed}: " "QML target-reach mismatch.")

    evaluations = payload["evaluations"]

    if not evaluations:
        raise ValueError("QML evaluations cannot be empty.")

    final = evaluations[-1]

    scientific_boundary = payload.get(
        "scientific_boundary",
        {},
    )

    if scientific_boundary:
        if scientific_boundary.get("quantum_hardware_used") is not False:
            raise ValueError("QML evidence must retain " "quantum_hardware_used=False.")

        if scientific_boundary.get("quantum_speedup_claimed") is not False:
            raise ValueError("QML quantum-speedup claim must remain blocked.")

    return QMLAblationRecord(
        domain=domain,
        policy="ppo_pqc",
        seed=seed,
        target_reached=target_reached,
        environment_steps_to_target=(None if qml_steps is None else int(qml_steps)),
        episodes_to_target=(None if qml_episodes is None else int(qml_episodes)),
        final_evaluation_reward=float(final["mean_reward"]),
        final_success_rate=float(final["success_rate"]),
        training_seconds=float(payload["training_seconds"]),
        actor_parameters=int(payload["actor_parameters"]),
        source_artifacts=(
            relative_path,
            ABLATION_SOURCE,
            PPO_TARGETS_SOURCE,
        ),
    )


def _build_matched_control(
    *,
    root: Path,
    domain: str,
    seed: int,
    ppo_targets: dict[str, Any],
    ablation: dict[str, Any],
) -> MatchedClassicalControlRecord:
    """Build one parameter-matched classical control record."""

    relative_path = _matched_run_path(
        domain=domain,
        seed=seed,
    )

    payload = _load_json(root / relative_path)

    if payload["policy"] != "matched_classical":
        raise ValueError("Expected matched_classical policy.")

    if payload["domain"] != domain:
        raise ValueError("Matched-classical domain mismatch.")

    if int(payload["seed"]) != seed:
        raise ValueError("Matched-classical seed mismatch.")

    target_entry = _find_seed_entry(
        ppo_targets["domains"][domain]["seeds"],
        seed=seed,
    )

    target_reward = float(target_entry["target_evaluation_reward"])

    reached, steps, episodes = _first_target_reach(
        evaluations=payload["evaluations"],
        target_reward=target_reward,
    )

    paired = _find_paired_record(
        ablation["paired_records"],
        domain=domain,
        seed=seed,
    )

    if bool(paired["matched_target_reached"]) != reached:
        raise ValueError(
            f"{domain}/seed-{seed}: " "matched-classical target-reach mismatch."
        )

    summary = payload["summary"]

    actor_parameters = int(payload["parameter_counts"]["actor_parameters"])

    expected_matched_parameters = int(paired["matched_actor_parameters"])

    if actor_parameters != expected_matched_parameters:
        raise ValueError(f"{domain}/seed-{seed}: " "matched actor parameter mismatch.")

    return MatchedClassicalControlRecord(
        domain=domain,
        seed=seed,
        target_reached=reached,
        environment_steps_to_target=steps,
        episodes_to_target=episodes,
        final_evaluation_reward=float(summary["final_mean_reward"]),
        final_success_rate=float(summary["final_success_rate"]),
        training_seconds=float(payload["training_seconds"]),
        actor_parameters=actor_parameters,
        source_artifacts=(
            relative_path,
            ABLATION_SOURCE,
            PPO_TARGETS_SOURCE,
        ),
    )


def load_canonical_qml_ablation(
    root: Path,
) -> CanonicalQMLAblation:
    """Load frozen Sprint 4 evidence for the Sprint 7.4 ablation."""

    ppo_summary = _load_json(root / PPO_SUMMARY_SOURCE)

    ppo_targets = _load_json(root / PPO_TARGETS_SOURCE)

    ablation = _load_json(root / ABLATION_SOURCE)

    validate_required_seeds(list(ppo_summary["seeds"]))

    validate_required_seeds(list(ablation["principal_seeds"]))

    if ppo_summary["method"] != "classical_ppo":
        raise ValueError("Expected classical_ppo PPO summary.")

    if ppo_targets["reference_method"] != "classical_ppo":
        raise ValueError("Expected classical_ppo target reference.")

    if ppo_targets["qml_results_seen"] is not False:
        raise ValueError("PPO targets must have been frozen before QML.")

    primary_records: list[QMLAblationRecord] = []

    controls: list[MatchedClassicalControlRecord] = []

    compactness: list[dict[str, Any]] = []

    for domain in DOMAINS:
        target_domain = ppo_targets["domains"][domain]

        if target_domain["targets_frozen_before_qml"] is not True:
            raise ValueError(f"{domain}: targets were not frozen before QML.")

        for seed in REQUIRED_SEEDS:
            primary_records.append(
                _build_ppo_record(
                    domain=domain,
                    seed=seed,
                    ppo_summary=ppo_summary,
                    ppo_targets=ppo_targets,
                )
            )

            primary_records.append(
                _build_qml_record(
                    root=root,
                    domain=domain,
                    seed=seed,
                    ppo_targets=ppo_targets,
                    ablation=ablation,
                )
            )

            controls.append(
                _build_matched_control(
                    root=root,
                    domain=domain,
                    seed=seed,
                    ppo_targets=ppo_targets,
                    ablation=ablation,
                )
            )

        ppo_method = ablation["domains"][domain]["methods"]["full_ppo"]

        qml_method = ablation["domains"][domain]["methods"]["qml"]

        compactness.append(
            compactness_record(
                domain=domain,
                classical_parameters=int(ppo_method["actor_parameters"]),
                qml_parameters=int(qml_method["actor_parameters"]),
                source_artifacts=(ABLATION_SOURCE,),
            )
        )

    expected_primary = {
        (
            domain,
            policy,
            seed,
        )
        for domain in DOMAINS
        for policy in POLICIES
        for seed in REQUIRED_SEEDS
    }

    observed_primary = {
        (
            record.domain,
            record.policy,
            record.seed,
        )
        for record in primary_records
    }

    if observed_primary != expected_primary:
        raise ValueError("Primary PPO/QML coverage is incomplete.")

    if len(controls) != 6:
        raise ValueError("Expected six matched-classical controls.")

    ppo_reach_count = sum(
        record.target_reached
        for record in primary_records
        if record.policy == "ppo_mlp"
    )

    qml_reach_count = sum(
        record.target_reached
        for record in primary_records
        if record.policy == "ppo_pqc"
    )

    matched_reach_count = sum(record.target_reached for record in controls)

    if ppo_reach_count != 6:
        raise ValueError("Frozen PPO target reach must remain 6/6.")

    if qml_reach_count != 0:
        raise ValueError("Frozen QML target reach must remain 0/6.")

    if matched_reach_count != 1:
        raise ValueError("Frozen matched-classical reach must remain 1/6.")

    return CanonicalQMLAblation(
        primary_records=tuple(primary_records),
        matched_classical_controls=tuple(controls),
        compactness=tuple(compactness),
    )
