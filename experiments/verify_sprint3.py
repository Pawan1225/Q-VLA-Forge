from __future__ import annotations

import hashlib
import json
from pathlib import Path

MANIFEST_PATH = Path("results") / "training" / "sprint3-training-manifest.json"


def _sha256(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def main() -> None:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(MANIFEST_PATH)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    print()
    print("===== Q-VLA Forge Sprint 3 Verification =====")

    print()
    print("[1] Run matrix")

    matrix = manifest["primary_run_matrix"]

    assert matrix["fp32_runs"] == 6

    assert matrix["structured_runs"] == 12

    assert matrix["total_runs"] == 18

    print("    PASS")

    print()
    print("[2] Controlled ablation")

    assert manifest["ablation"]["runs"] == 4

    assert manifest["ablation"]["performance_blind_pair_selection"] is True

    print("    PASS")

    print()
    print("[3] Training protocol")

    protocol = manifest["protocol"]

    assert protocol["seeds"] == [
        42,
        123,
        456,
    ]

    assert protocol["epochs"] == 20

    assert protocol["batch_size"] == 32

    assert protocol["robust_efficiency_threshold_percent"] == 10.0

    print("    PASS")

    print()
    print("[4] Scientific claim controls")

    quantum = manifest["quantum_claim_control"]

    assert quantum["tt_mps_is_quantum_inspired"] is True

    assert quantum["quantum_hardware_used"] is False

    assert quantum["quantum_advantage_claimed"] is False

    assert quantum["quantum_speedup_claimed"] is False

    assert quantum["native_tt_runtime_speedup_claimed"] is False

    print("    PASS")

    print()
    print("[5] Architecture claim controls")

    architecture = manifest["architecture_claim_control"]

    assert architecture["shared_architecture"] is True

    assert architecture["universal_trained_model"] is False

    assert architecture["language_component"] == "lightweight trainable text encoder"

    print("    PASS")

    print()
    print("[6] Artifact integrity")

    checked = 0

    for artifact in manifest["artifacts"]:
        path = Path(artifact["path"])

        if not path.exists():
            raise FileNotFoundError(path)

        observed = _sha256(path)

        expected = artifact["sha256"]

        if observed != expected:
            raise RuntimeError(f"hash mismatch: {path}")

        checked += 1

    print(f"    PASS ({checked} artifacts)")

    print()
    print("[7] Scientific run total")

    assert manifest["total_scientific_training_runs"] == 22

    print("    PASS")

    print()
    print("============================================")
    print(" SPRINT 3 SCIENTIFIC VERIFICATION PASSED")
    print("============================================")


if __name__ == "__main__":
    main()
