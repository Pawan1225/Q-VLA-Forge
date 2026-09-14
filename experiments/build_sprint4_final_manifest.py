"""Build the final Sprint 4 scientific freeze manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from q_vla_forge.evaluation.rl_final_gate import (
    FrozenArtifact,
    artifact_to_dict,
    build_frozen_artifact,
)

OUTPUT_PATH: Final = Path("results/rl/final/sprint4-final-manifest.json")

PRINCIPAL_SEEDS: Final = (
    42,
    123,
    456,
)

ArtifactSpec = tuple[
    str,
    str,
    str,
    str,
]


def canonical_specs() -> list[ArtifactSpec]:
    """Return the fixed canonical Sprint 4 artifact inventory."""

    return [
        (
            "results/rl/sprint4-rl-protocol.json",
            "4.1",
            "protocol",
            "rl_protocol",
        ),
        (
            "src/q_vla_forge/rl/driving_env.py",
            "4.2",
            "environments",
            "driving_environment",
        ),
        (
            "src/q_vla_forge/rl/robotics_env.py",
            "4.3",
            "environments",
            "robotics_environment",
        ),
        (
            "results/rl/environment-audit/sprint4-environment-audit.json",
            "4.4",
            "environment_audit",
            "environment_audit",
        ),
        (
            "results/rl/ppo/sprint4-ppo-summary.json",
            "4.5",
            "ppo",
            "ppo_summary",
        ),
        (
            "results/rl/ppo/sprint4-ppo-targets.json",
            "4.5",
            "targets",
            "frozen_paired_targets",
        ),
        (
            "results/rl/qml-foundation/sprint4-qml-foundation.json",
            "4.6",
            "qml_foundation",
            "pqc_foundation",
        ),
        (
            "results/rl/hybrid-policy/sprint4-hybrid-qml-policy.json",
            "4.7",
            "hybrid_policy",
            "hybrid_qml_policy",
        ),
        (
            "results/rl/qml/sprint4-driving-qml-summary.json",
            "4.8",
            "driving_qml",
            "driving_qml_summary",
        ),
        (
            "results/rl/qml/sprint4-robotics-qml-summary.json",
            "4.9",
            "robotics_qml",
            "robotics_qml_summary",
        ),
        (
            "results/rl/validation/sprint4-three-seed-validation.json",
            "4.10",
            "validation",
            "three_seed_validation",
        ),
        (
            "results/rl/analysis/sprint4-sample-efficiency.json",
            "4.11",
            "sample_efficiency",
            "sample_efficiency_analysis",
        ),
        (
            "results/rl/ablation/sprint4-classical-vs-qml-ablation.json",
            "4.12",
            "ablation",
            "matched_budget_ablation",
        ),
        (
            "results/rl/cross-domain/sprint4-cross-domain.json",
            "4.13",
            "cross_domain",
            "cross_domain_analysis",
        ),
        (
            "results/rl/evidence/sprint4-rl-evidence.json",
            "4.14",
            "proposal_evidence",
            "consolidated_evidence_json",
        ),
        (
            "results/rl/evidence/sprint4-rl-evidence.csv",
            "4.14",
            "proposal_evidence",
            "consolidated_evidence_csv",
        ),
        (
            "results/rl/evidence/sprint4-rl-evidence.md",
            "4.14",
            "proposal_evidence",
            "proposal_evidence_markdown",
        ),
        (
            "results/rl/evidence/sprint4-figure-index.json",
            "4.14",
            "figures",
            "figure_index",
        ),
        (
            "results/rl/evidence/sprint4-claim-matrix.json",
            "4.14",
            "claims",
            "claim_matrix",
        ),
        (
            "results/rl/evidence/sprint4-manifest.json",
            "4.14",
            "proposal_evidence",
            "proposal_manifest",
        ),
        (
            "results/pilot-readiness/claim-registry.json",
            "4.14",
            "claims",
            "claim_registry",
        ),
        (
            "dashboard/data_loader.py",
            "4.14",
            "dashboard",
            "dashboard_loader",
        ),
        (
            "dashboard/pages/4_RL_Hybrid_QML.py",
            "4.14",
            "dashboard",
            "dashboard_page",
        ),
    ]


def raw_run_specs() -> list[ArtifactSpec]:
    """Return all frozen principal and reproduction run artifacts."""

    specs: list[ArtifactSpec] = []

    for seed in PRINCIPAL_SEEDS:
        specs.extend(
            [
                (
                    f"results/rl/ppo/autonomous_driving-seed-{seed}.json",
                    "4.5",
                    "ppo",
                    f"driving_ppo_seed_{seed}",
                ),
                (
                    f"results/rl/ppo/robotics-seed-{seed}.json",
                    "4.5",
                    "ppo",
                    f"robotics_ppo_seed_{seed}",
                ),
                (
                    f"results/rl/qml/autonomous_driving-seed-{seed}.json",
                    "4.8",
                    "driving_qml",
                    f"driving_qml_seed_{seed}",
                ),
                (
                    f"results/rl/qml/robotics-seed-{seed}.json",
                    "4.9",
                    "robotics_qml",
                    f"robotics_qml_seed_{seed}",
                ),
                (
                    (
                        "results/rl/ablation/"
                        f"matched-classical-autonomous_driving-seed-{seed}.json"
                    ),
                    "4.12",
                    "ablation",
                    f"driving_matched_classical_seed_{seed}",
                ),
                (
                    (
                        "results/rl/ablation/"
                        f"matched-classical-robotics-seed-{seed}.json"
                    ),
                    "4.12",
                    "ablation",
                    f"robotics_matched_classical_seed_{seed}",
                ),
            ]
        )

    specs.extend(
        [
            (
                "results/rl/qml/autonomous_driving-seed-42-repro.json",
                "4.8",
                "driving_qml",
                "driving_qml_seed_42_reproduction",
            ),
            (
                "results/rl/qml/robotics-seed-42-repro.json",
                "4.9",
                "robotics_qml",
                "robotics_qml_seed_42_reproduction",
            ),
            (
                (
                    "results/rl/ablation/"
                    "matched-classical-autonomous_driving-seed-42-repro.json"
                ),
                "4.12",
                "ablation",
                "driving_matched_seed_42_reproduction",
            ),
            (
                (
                    "results/rl/ablation/"
                    "matched-classical-robotics-seed-42-repro.json"
                ),
                "4.12",
                "ablation",
                "robotics_matched_seed_42_reproduction",
            ),
        ]
    )

    return specs


def figure_specs() -> list[ArtifactSpec]:
    """Return the six proposal-relevant Sprint 4 figures."""

    return [
        (
            "figures/rl/driving-normalized-progress.png",
            "4.11",
            "figures",
            "driving_normalized_progress",
        ),
        (
            "figures/rl/robotics-normalized-progress.png",
            "4.11",
            "figures",
            "robotics_normalized_progress",
        ),
        (
            "figures/rl/driving-matched-ablation.png",
            "4.12",
            "figures",
            "driving_matched_ablation",
        ),
        (
            "figures/rl/robotics-matched-ablation.png",
            "4.12",
            "figures",
            "robotics_matched_ablation",
        ),
        (
            "figures/rl/cross-domain-matched-delta.png",
            "4.13",
            "figures",
            "cross_domain_matched_delta",
        ),
        (
            "figures/rl/cross-domain-parameter-compactness.png",
            "4.13",
            "figures",
            "cross_domain_parameter_compactness",
        ),
    ]


def all_specs() -> list[ArtifactSpec]:
    """Return the complete deterministic freeze specification."""

    specs = canonical_specs() + raw_run_specs() + figure_specs()

    paths = [path for path, _, _, _ in specs]

    if len(paths) != len(set(paths)):
        duplicates = sorted({path for path in paths if paths.count(path) > 1})

        raise RuntimeError(f"Duplicate manifest paths: {duplicates}")

    return specs


def build_manifest() -> dict[str, object]:
    """Build the complete Sprint 4 freeze manifest."""

    specs = all_specs()

    artifact_records: list[tuple[str, FrozenArtifact]] = []

    group_paths: dict[
        str,
        list[str],
    ] = {}

    for (
        path_str,
        sprint,
        group,
        role,
    ) in specs:
        artifact = build_frozen_artifact(
            path=Path(path_str),
            sprint=sprint,
            role=role,
            required=True,
            frozen=True,
        )

        artifact_records.append(
            (
                group,
                artifact,
            )
        )

        group_paths.setdefault(
            group,
            [],
        ).append(artifact.path)

    artifacts = [artifact_to_dict(artifact) for _, artifact in artifact_records]

    groups = {group: sorted(paths) for group, paths in sorted(group_paths.items())}

    return {
        "sprint": "4",
        "freeze_sprint": "4.15",
        "status": "frozen",
        "principal_seeds": list(PRINCIPAL_SEEDS),
        "artifact_count": len(artifacts),
        "group_count": len(groups),
        "groups": groups,
        "artifacts": artifacts,
    }


def main() -> None:
    """Build and save the final manifest."""

    manifest = build_manifest()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "Sprint 4 final manifest:",
        OUTPUT_PATH.as_posix(),
    )
    print(
        "Artifacts:",
        manifest["artifact_count"],
    )
    print(
        "Groups:",
        manifest["group_count"],
    )


if __name__ == "__main__":
    main()
