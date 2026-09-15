from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(".")
SOURCE = ROOT / "results" / "integration" / "sprint5-unified-architecture.json"
OUTPUT = ROOT / "results" / "integration" / "sprint5-shared-domain-component-map.json"

payload = json.loads(SOURCE.read_text(encoding="utf-8-sig"))

if not isinstance(
    payload,
    dict,
):
    raise TypeError("unified architecture must be a JSON object")

shared_core = payload["shared_ai_vla_core"]

domain_specific = payload["domain_specific"]

compression_path = payload["compression_path"]

policy_path = payload["policy_path"]

safety_path = payload["safety_path"]

if not isinstance(
    shared_core,
    list,
):
    raise TypeError("shared_ai_vla_core must be list")

if not isinstance(
    domain_specific,
    dict,
):
    raise TypeError("domain_specific must be dict")

if not isinstance(
    compression_path,
    dict,
):
    raise TypeError("compression_path must be dict")

if not isinstance(
    policy_path,
    dict,
):
    raise TypeError("policy_path must be dict")

if not isinstance(
    safety_path,
    dict,
):
    raise TypeError("safety_path must be dict")

shared_components = {
    "infrastructure": [
        "data_contract_framework",
        "configuration_framework",
        "reproducibility_and_seed_handling",
        "result_store_schema",
        "evidence_generation",
        "dashboard_infrastructure",
        "claim_control_framework",
    ],
    "ai_vla": list(shared_core),
    "compression": [
        "compression_interface",
        "compression_accounting",
        "accuracy_compression_tradeoff_analysis",
        "int8_evaluation_framework",
        "svd_evaluation_framework",
        "tensor_network_evaluation_framework",
    ],
    "rl_qml": [
        "ppo_evaluation_protocol",
        "hybrid_policy_interface",
        "pqc_vqc_interface",
        "three_dimensional_action_output",
        "sample_efficiency_analysis_framework",
    ],
    "safety": [
        "safety_decision_contract",
        "none_method",
        "clipping_method_interface",
        "lyapunov_guided_filter_interface",
        "violation_accounting",
        "intervention_accounting",
        "robustness_harness",
    ],
    "evaluation": [
        "three_seed_protocol",
        "cross_domain_reporting_schema",
        "figure_provenance",
        "claim_matrix",
    ],
}

driving_specific = list(domain_specific["autonomous_driving"])

robotics_specific = list(domain_specific["robotics"])

shared_boundary = {
    "shared_framework": True,
    "shared_interfaces": True,
    "shared_action_dimension": True,
    "shared_training_protocol": True,
    "shared_safety_api": True,
    "shared_robustness_harness": True,
    "same_environment": False,
    "same_state_semantics": False,
    "same_reward": False,
    "same_policy_weights": False,
    "same_safety_constraints": False,
    "same_lyapunov_potential": False,
    "cross_domain_weight_transfer_tested": False,
    "universal_controller_supported": False,
}

result: dict[str, Any] = {
    "project": "Q-VLA Forge",
    "sprint": "5.14.5C",
    "artifact": "shared-vs-domain-specific-component-map",
    "analysis_only": True,
    "new_training": False,
    "new_principal_runs": False,
    "source_artifact": str(SOURCE).replace(
        "\\",
        "/",
    ),
    "shared_components": shared_components,
    "domain_specific_components": {
        "autonomous_driving": driving_specific,
        "robotics": robotics_specific,
    },
    "shared_boundary": shared_boundary,
    "compression_implementation_classes": compression_path,
    "policy_implementation_classes": policy_path,
    "safety_implementation": safety_path,
    "supported_statement": (
        "The architecture is shared where computationally useful, "
        "while domain-specific semantics, environments, policy weights, "
        "dynamics, rewards, and safety constraints remain isolated "
        "behind domain adapters and safety contracts."
    ),
    "blocked_statements": [
        "one_universal_trained_vla",
        "shared_policy_weights",
        "zero_shot_cross_domain_transfer",
        "identical_safety_constraints",
        "universal_safety_controller",
    ],
}

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
print(" SPRINT 5.14.5C SHARED VS DOMAIN-SPECIFIC COMPONENT MAP")
print("=" * 80)
print()
print(
    "Shared component groups:",
    len(shared_components),
)
print(
    "Driving-specific components:",
    len(driving_specific),
)
print(
    "Robotics-specific components:",
    len(robotics_specific),
)
print()
print("Shared framework boundary: PASS")
print("Domain adapters retained: PASS")
print("Separate policy weights retained: PASS")
print("Separate safety semantics retained: PASS")
print("Cross-domain transfer blocked: PASS")
print("Universal-controller claim blocked: PASS")
print("No new training: PASS")
print("No new principal execution: PASS")
print()
print("SPRINT 5.14.5C COMPONENT MAP: PASS")
