"""Sprint 7.6 — final frozen RL/QML ablation summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RL_QML_ABLATION = "results/rl/ablation/" "sprint4-classical-vs-qml-ablation.json"

EXPECTED_SEEDS = (
    42,
    123,
    456,
)

EXPECTED_DOMAINS = (
    "autonomous_driving",
    "robotics",
)


def load_json(
    path: Path,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")

    return payload


def parameter_reduction_percent(
    reference: int,
    candidate: int,
) -> float:
    if reference <= 0:
        raise ValueError("Reference parameter count must be positive.")

    return (1.0 - candidate / reference) * 100.0


def build_final_rl_qml(
    root: Path,
) -> dict[str, Any]:
    """Build the Sprint 7.6 frozen RL/QML comparison."""

    source_path = root / RL_QML_ABLATION

    if not source_path.exists():
        raise FileNotFoundError(source_path)

    source = load_json(source_path)

    principal_seeds = tuple(
        source.get(
            "principal_seeds",
            [],
        )
    )

    if principal_seeds != EXPECTED_SEEDS:
        raise ValueError("RL/QML ablation does not use " "locked seeds 42, 123, 456.")

    domains_source = source.get("domains")

    if not isinstance(
        domains_source,
        dict,
    ):
        raise KeyError("Missing RL/QML domain results.")

    domains: dict[str, Any] = {}

    qml_target_reached_total = 0
    ppo_target_reached_total = 0
    matched_target_reached_total = 0

    for domain in EXPECTED_DOMAINS:
        domain_payload = domains_source.get(domain)

        if not isinstance(
            domain_payload,
            dict,
        ):
            raise KeyError(f"Missing RL/QML domain: {domain}")

        methods = domain_payload.get("methods")

        if not isinstance(
            methods,
            dict,
        ):
            raise KeyError(f"Missing methods for {domain}")

        ppo = methods["full_ppo"]
        matched = methods["matched_classical"]
        qml = methods["qml"]

        ppo_actor = int(ppo["actor_parameters"])

        qml_actor = int(qml["actor_parameters"])

        matched_actor = int(matched["actor_parameters"])

        qml_reduction = parameter_reduction_percent(
            ppo_actor,
            qml_actor,
        )

        matched_reduction = parameter_reduction_percent(
            ppo_actor,
            matched_actor,
        )

        ppo_target_reached_total += int(ppo["target_reach_count"])

        matched_target_reached_total += int(matched["target_reach_count"])

        qml_target_reached_total += int(qml["target_reach_count"])

        domains[domain] = {
            "full_ppo": ppo,
            "matched_classical": matched,
            "qml": qml,
            "parameter_reduction": {
                "qml_actor_vs_ppo_percent": (qml_reduction),
                "matched_actor_vs_ppo_percent": (matched_reduction),
            },
            "paired_delta_summary": (
                domain_payload.get(
                    "paired_delta_summary",
                    {},
                )
            ),
        }

    qml_parameter_reduction = {
        domain: domains[domain]["parameter_reduction"]["qml_actor_vs_ppo_percent"]
        for domain in EXPECTED_DOMAINS
    }

    return {
        "sprint": "7.6",
        "protocol": ("final_rl_qml_ablation"),
        "status": "FROZEN",
        "source": RL_QML_ABLATION,
        "required_seeds": list(EXPECTED_SEEDS),
        "parameter_matched_ablation": bool(
            source.get(
                "parameter_matched_ablation",
                False,
            )
        ),
        "domains": domains,
        "aggregate_target_reach": {
            "full_ppo": (ppo_target_reached_total),
            "matched_classical": (matched_target_reached_total),
            "qml": (qml_target_reached_total),
            "available_per_method": 6,
        },
        "qml_actor_parameter_reduction_vs_ppo_percent": (qml_parameter_reduction),
        "qml_sample_efficiency_advantage_demonstrated": (False),
        "qml_computational_advantage_demonstrated": (False),
        "claim": (
            "The evaluated PQC/QML actor provides a "
            "large actor-parameter reduction relative "
            "to full PPO, but Phase 1 does not "
            "demonstrate QML sample-efficiency or "
            "computational advantage."
        ),
        "scientific_boundary": source.get(
            "scientific_boundary",
            {},
        ),
        "new_training": False,
    }
