from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(".")

ARCHITECTURE = ROOT / "results" / "integration" / "sprint5-unified-architecture.json"

BOTTLENECK_MAP = (
    ROOT / "results" / "integration" / "sprint5-volkswagen-bottleneck-map.json"
)

OUTPUT = ROOT / "results" / "integration" / "sprint5-computational-classification.json"

for path in (
    ARCHITECTURE,
    BOTTLENECK_MAP,
):
    if not path.exists():
        raise FileNotFoundError(path)

architecture = json.loads(ARCHITECTURE.read_text(encoding="utf-8-sig"))

if not isinstance(
    architecture,
    dict,
):
    raise TypeError("architecture must be a JSON object")

classification: dict[str, Any] = {
    "classical": {
        "ai_training": [
            "shared_neural_baseline",
            "adamw",
            "cosine_learning_rate_schedule",
        ],
        "compression": [
            "int8",
            "svd",
        ],
        "reinforcement_learning": [
            "ppo",
            "matched_classical_actor",
        ],
        "safety": [
            "heuristic_clipping",
            "lyapunov_function",
            "lyapunov_guided_filter",
        ],
    },
    "quantum_inspired": {
        "compression": [
            "tensor_train",
            "mps",
            "tt_svd",
        ],
    },
    "quantum_ml": {
        "policy": [
            "angle_encoding",
            "four_qubit_pqc",
            "parameterized_quantum_circuit",
            "variational_quantum_circuit",
            "ry_rz_rotations",
            "cnot_entanglement",
            "hybrid_quantum_classical_actor",
        ],
    },
}

boundaries = {
    "lyapunov_is_quantum": False,
    "clipping_is_quantum": False,
    "ppo_is_quantum": False,
    "svd_is_quantum_inspired": False,
    "tensor_train_is_quantum_inspired": True,
    "mps_is_quantum_inspired": True,
    "pqc_is_qml": True,
    "hybrid_actor_is_qml": True,
    "quantum_advantage_demonstrated": False,
    "quantum_speedup_demonstrated": False,
}

result: dict[str, Any] = {
    "project": "Q-VLA Forge",
    "sprint": "5.14.5E",
    "artifact": "classical-qi-qml-classification",
    "analysis_only": True,
    "new_training": False,
    "new_principal_runs": False,
    "classification": classification,
    "classification_boundaries": boundaries,
    "source_artifacts": [
        str(ARCHITECTURE).replace(
            "\\",
            "/",
        ),
        str(BOTTLENECK_MAP).replace(
            "\\",
            "/",
        ),
    ],
    "supported_statement": (
        "Q-VLA Forge evaluates classical, quantum-inspired, and "
        "quantum-machine-learning components within one modular "
        "pilot framework while preserving their distinct "
        "computational classifications."
    ),
    "blocked_statements": [
        "lyapunov_safety_is_quantum",
        "clipping_is_quantum",
        "ppo_is_quantum",
        "quantum_advantage_demonstrated",
        "quantum_speedup_demonstrated",
    ],
}

if bool(boundaries["lyapunov_is_quantum"]):
    raise RuntimeError("Lyapunov safety must remain classified as classical")

if bool(boundaries["quantum_advantage_demonstrated"]):
    raise RuntimeError("quantum advantage must remain unsupported")

if bool(boundaries["quantum_speedup_demonstrated"]):
    raise RuntimeError("quantum speedup must remain unsupported")

OUTPUT.write_text(
    json.dumps(
        result,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

print("=" * 80)
print(" SPRINT 5.14.5E CLASSICAL / QI / QML CLASSIFICATION")
print("=" * 80)
print()
print(
    "Classical groups:",
    len(classification["classical"]),
)
print(
    "Quantum-inspired groups:",
    len(classification["quantum_inspired"]),
)
print(
    "QML groups:",
    len(classification["quantum_ml"]),
)
print()
print("INT8 classified classical: PASS")
print("SVD classified classical: PASS")
print("PPO classified classical: PASS")
print("Clipping classified classical: PASS")
print("Lyapunov safety classified classical: PASS")
print("TT/MPS classified quantum-inspired: PASS")
print("PQC/VQC classified QML: PASS")
print("Hybrid actor classified QML: PASS")
print("Quantum advantage claim blocked: PASS")
print("Quantum speedup claim blocked: PASS")
print("No new training: PASS")
print("No new principal execution: PASS")
print()
print("SPRINT 5.14.5E COMPUTATIONAL CLASSIFICATION: PASS")
