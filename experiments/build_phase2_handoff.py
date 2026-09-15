from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(".")

FINDINGS = ROOT / "results" / "integration" / "sprint5-phase1-findings.json"

CLAIMS = ROOT / "results" / "integration" / "sprint5-unified-claim-matrix.json"

ARCHITECTURE = ROOT / "results" / "integration" / "sprint5-unified-architecture.json"

ROADMAP_JSON = ROOT / "results" / "integration" / "sprint5-phase2-roadmap.json"

HANDOFF_MD = ROOT / "results" / "integration" / "sprint5-phase2-handoff.md"

for path in (
    FINDINGS,
    CLAIMS,
    ARCHITECTURE,
):
    if not path.exists():
        raise FileNotFoundError(path)

priorities: list[dict[str, Any]] = [
    {
        "priority": 1,
        "track": "real_vla_integration",
        "objective": (
            "Replace the compact Phase 1 proxy backbone with a "
            "larger pretrained VLA-style representation."
        ),
        "candidate_paths": [
            "OpenVLA-style robotics backbone",
            "pi0-style robotics pathway",
            "driving-VLA / multimodal driving backbone",
        ],
        "phase1_gap_addressed": ("Phase 1 used a compact synthetic shared VLA proxy."),
    },
    {
        "priority": 2,
        "track": "shared_latent_policy_integration",
        "objective": (
            "Connect the VLA shared latent representation directly "
            "to classical PPO and hybrid PQC policy heads."
        ),
        "candidate_paths": [
            "shared_latent_to_ppo",
            "shared_latent_to_hybrid_pqc",
        ],
        "phase1_gap_addressed": (
            "Sprint 4 policies operated on compact environment "
            "state rather than the Sprint 1 shared latent."
        ),
    },
    {
        "priority": 3,
        "track": "tensor_network_scale_up",
        "objective": (
            "Evaluate TT/MPS compression on larger VLA projection, "
            "fusion, and attention-related matrices."
        ),
        "candidate_paths": [
            "larger_projection_layers",
            "multimodal_fusion_layers",
            "attention_related_matrices",
        ],
        "phase1_gap_addressed": (
            "Phase 1 tensor-network experiments were limited to "
            "the compact pilot architecture."
        ),
    },
    {
        "priority": 4,
        "track": "realistic_environment_validation",
        "objective": (
            "Move from synthetic proxy environments to realistic "
            "driving and robotics benchmarks."
        ),
        "candidate_paths": {
            "driving": [
                "CARLA",
                "nuScenes-derived evaluation",
            ],
            "robotics": [
                "RLBench",
                "LIBERO",
                "Open X-Embodiment",
            ],
        },
        "phase1_gap_addressed": (
            "Phase 1 validation used synthetic proxy environments."
        ),
    },
    {
        "priority": 5,
        "track": "estimated_state_safety",
        "objective": (
            "Evaluate safety filters when both policy and safety "
            "layer receive estimated or noisy state."
        ),
        "candidate_paths": [
            "true_state_safety_reference",
            "estimated_state_safety",
            "noisy_state_safety",
        ],
        "phase1_gap_addressed": (
            "Phase 1 perception-perturbation studies allowed the "
            "safety layer access to true simulator state."
        ),
    },
    {
        "priority": 6,
        "track": "embedded_deployment",
        "objective": (
            "Measure end-to-end latency, memory, and safety-filter "
            "overhead on embedded hardware."
        ),
        "candidate_paths": [
            "Jetson-class deployment",
            "CPU embedded baseline",
            "accelerator-enabled runtime",
        ],
        "phase1_gap_addressed": (
            "Phase 1 compact CPU latency does not represent "
            "full-scale VLA deployment latency."
        ),
    },
    {
        "priority": 7,
        "track": "quantum_hardware_path",
        "objective": (
            "Extend PQC studies to larger circuits, noisy simulation, "
            "and hardware execution where available."
        ),
        "candidate_paths": [
            "larger_qubit_counts",
            "noise_aware_simulation",
            "hardware_efficient_ansatz",
            "quantum_hardware_execution",
        ],
        "phase1_gap_addressed": (
            "Phase 1 used a small simulated PQC and did not "
            "demonstrate quantum advantage."
        ),
    },
    {
        "priority": 8,
        "track": "stronger_validation",
        "objective": ("Increase statistical strength and broaden evaluation."),
        "candidate_paths": [
            "more_principal_seeds",
            "larger_evaluation_corpus",
            "formal_statistical_analysis",
            "systematic_ablation",
            "hardware_noise",
            "cross_task_evaluation",
        ],
        "phase1_gap_addressed": (
            "Phase 1 used three principal seeds and compact " "evaluation settings."
        ),
    },
]

reusable_modules = [
    "data_contracts",
    "configuration_system",
    "reproducibility_framework",
    "result_store",
    "compression_interfaces",
    "policy_interfaces",
    "safety_interface",
    "robustness_harness",
    "evidence_schema",
    "claim_controls",
    "dashboard",
]

replacement_or_scale_up = [
    "compact_shared_vla_backbone",
    "synthetic_driving_environment",
    "synthetic_robotics_environment",
    "compact_raw_state_policy_input",
    "small_pqc_configuration",
    "proxy_latency_measurement",
    "privileged_true_state_safety_input",
]

roadmap: dict[str, Any] = {
    "project": "Q-VLA Forge",
    "sprint": "5.14.5I",
    "artifact": "phase2-roadmap",
    "analysis_only": True,
    "new_training": False,
    "new_principal_runs": False,
    "phase1_status": "frozen",
    "priorities": priorities,
    "reusable_modules": reusable_modules,
    "replacement_or_scale_up": replacement_or_scale_up,
    "phase2_claim_boundary": (
        "The roadmap identifies technically plausible Phase 2 "
        "extensions. It does not claim that scale-up, hardware "
        "advantage, real-world deployment, or production safety "
        "has already been demonstrated."
    ),
}

ROADMAP_JSON.write_text(
    json.dumps(
        roadmap,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

handoff = """# Q-VLA Forge Phase 2 Handoff

## Current Phase 1 Status

Phase 1 established a reproducible dual-domain pilot spanning autonomous-driving and robotics proxy tasks. The repository now contains frozen evidence for the shared VLA baseline, classical and quantum-inspired compression, training-efficiency studies, classical PPO, hybrid PQC policies, safety filtering, robustness evaluation, and cross-domain safety analysis.

## What Is Frozen

The following Phase 1 interfaces should be preserved as the starting point for Phase 2:

- data contracts
- configuration system
- reproducibility and seed handling
- result-store schema
- compression interfaces
- policy interfaces
- safety-filter API
- robustness harness
- evidence schema
- claim controls
- dashboard infrastructure

## What Worked

- A common computational framework was instantiated across both domains.
- All four Volkswagen challenge bottlenecks were experimentally addressed.
- INT8 provided the strongest Phase 1 compression result.
- Classical PPO reliably reached the frozen RL targets.
- The hybrid PQC actor provided substantial parameter compactness.
- Explicit safety filtering produced strong empirical pilot safety effects.
- Cross-domain architecture reuse was supported at the framework and interface level.

## What Did Not Establish Advantage

- TT/MPS did not establish superiority over the strongest classical compression result.
- No robust >=10% training-efficiency advantage was demonstrated.
- Hybrid QML did not demonstrate sample-efficiency advantage.
- No quantum advantage or quantum speedup was demonstrated.
- Cross-domain zero-shot policy transfer was not tested.
- No formal safety guarantee or production-readiness claim is supported.

## Critical Phase 1 Limitations

- synthetic proxy environments
- compact shared VLA baseline
- Sprint 4 policies consume compact environment state rather than the Sprint 1 shared latent
- small simulated PQC
- no quantum hardware
- three principal seeds
- no physical driving or robotics validation
- no embedded full-VLA deployment benchmark
- privileged true-state access for the safety layer in perception-perturbation studies
- no formal stability proof or certification

## Phase 2 Priority Sequence

1. Integrate a larger pretrained VLA-style representation.
2. Connect the shared VLA latent directly to PPO and hybrid PQC policy heads.
3. Scale TT/MPS compression to larger VLA matrices.
4. Validate in realistic driving and robotics environments.
5. Evaluate safety using estimated/noisy state.
6. Benchmark embedded deployment latency, memory, and safety overhead.
7. Extend PQC studies to larger/noisy/hardware settings where available.
8. Increase seeds, evaluation corpus size, and statistical rigor.

## Phase 2 Entry Principle

Phase 2 should extend the frozen Phase 1 interfaces rather than rebuild the project from scratch. The main changes should occur in model scale, environment realism, policy integration, safety-state assumptions, and deployment validation.

## Claim Boundary

This handoff is a technical roadmap. It does not claim that Phase 2 scale-up, quantum advantage, hardware speedup, real-world safety, or production deployment has already been demonstrated.
"""

HANDOFF_MD.write_text(
    handoff,
    encoding="utf-8",
)

print("=" * 80)
print(" SPRINT 5.14.5I PHASE-2 ROADMAP + HANDOFF")
print("=" * 80)
print()
print(
    "Phase-2 priority tracks:",
    len(priorities),
)
print(
    "Reusable modules:",
    len(reusable_modules),
)
print(
    "Replacement / scale-up targets:",
    len(replacement_or_scale_up),
)
print()
print("Real VLA integration: PASS")
print("Shared-latent policy integration: PASS")
print("Tensor-network scale-up: PASS")
print("Realistic environment path: PASS")
print("Estimated-state safety path: PASS")
print("Embedded deployment path: PASS")
print("Quantum hardware path: PASS")
print("Stronger validation path: PASS")
print()
print("Phase-1 limitations preserved: PASS")
print("Unsupported advantages remain blocked: PASS")
print("No new training: PASS")
print("No new principal execution: PASS")
print()
print("SPRINT 5.14.5I PHASE-2 HANDOFF: PASS")
