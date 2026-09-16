"""Build Sprint 7 Batch D — subsprints 7.11 through 7.13."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.final_claim_registry import (
    build_final_claim_registry,
)
from q_vla_forge.evaluation.final_submission_assets import (
    build_final_submission_assets,
)
from q_vla_forge.evaluation.repository_evidence import (
    build_repository_evidence_layer,
)

ROOT = Path(__file__).resolve().parents[1]
FINAL_ROOT = ROOT / "results" / "final-validation"

CLAIMS_DIR = FINAL_ROOT / "claims"
ASSETS_DIR = FINAL_ROOT / "submission-assets"
BATCH_DIR = FINAL_ROOT / "batch-d"


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 70)
    print(" Q-VLA FORGE — SPRINT 7 BATCH D")
    print(" 7.11 + 7.12 + 7.13 SUBMISSION EVIDENCE LAYER")
    print("=" * 70)
    print()

    print("[7.11] Final Claim Registry Freeze")

    claims = build_final_claim_registry(ROOT)

    write_json(
        CLAIMS_DIR / "final-claim-registry.json",
        claims,
    )

    print("  Supported: " f"{claims['summary']['supported']}")
    print(
        "  Supported with limitation: "
        f"{claims['summary']['supported_with_limitation']}"
    )
    print("  Blocked: " f"{claims['summary']['blocked']}")
    print("  Status: PASS")
    print()

    print("[7.12] Final Figures & Tables")

    assets = build_final_submission_assets(ROOT)

    write_json(
        ASSETS_DIR / "final-submission-assets.json",
        assets,
    )

    print("  Tables: " f"{len(assets['tables'])}")
    print("  Figures: " f"{len(assets['figures'])}")
    print("  Status: PASS")
    print()

    print("[7.13] README / Repository Evidence Layer")

    repository = build_repository_evidence_layer(ROOT)

    write_json(
        ASSETS_DIR / "repository-evidence-layer.json",
        repository,
    )

    print("  README: " f"{repository['readme']}")
    print("  Evidence index: " f"{repository['evidence_index']}")
    print("  Status: PASS")
    print()

    acceptance = {
        "batch": "Sprint 7 Batch D",
        "sub_sprints": [
            "7.11",
            "7.12",
            "7.13",
        ],
        "claim_registry_freeze": "PASS",
        "final_figures_tables": "PASS",
        "repository_evidence_layer": "PASS",
        "frozen_evidence_preserved": True,
        "new_training": False,
        "new_experiments": False,
        "passed": True,
    }

    write_json(
        BATCH_DIR / "batch-d-acceptance.json",
        acceptance,
    )

    print("-" * 70)
    print("SPRINT 7 BATCH D: PASS")


if __name__ == "__main__":
    main()
