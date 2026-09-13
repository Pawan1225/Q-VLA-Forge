"""Generate the frozen Sprint 4 RL experimental protocol."""

from __future__ import annotations

import json
from pathlib import Path

from q_vla_forge.rl.protocol import protocol_to_dict

OUTPUT = Path("results") / "rl" / "sprint4-rl-protocol.json"


def main() -> None:
    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = protocol_to_dict()

    OUTPUT.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("==============================================")
    print(" Q-VLA FORGE — SPRINT 4 RL PROTOCOL")
    print("==============================================")

    print(
        "Domains:",
        ", ".join(payload["domains"]),
    )

    print(
        "Seeds:",
        payload["seeds"],
    )

    print(
        "Episode horizon:",
        payload["episode_horizon"],
    )

    print(
        "Environment-step budget:",
        payload["maximum_environment_steps"],
    )

    print(
        "Evaluation frequency:",
        payload["evaluation_frequency_steps"],
    )

    print(
        "Evaluation episodes:",
        payload["evaluation_episodes"],
    )

    print(
        "Primary metric:",
        payload["primary_efficiency_metric"],
    )

    print(
        "QML qubits:",
        payload["qml"]["qubits"],
    )

    print(
        "QML execution:",
        payload["qml"]["execution"],
    )

    print(
        "Quantum hardware:",
        payload["qml"]["quantum_hardware_used"],
    )

    print(
        "Protocol:",
        OUTPUT,
    )


if __name__ == "__main__":
    main()
