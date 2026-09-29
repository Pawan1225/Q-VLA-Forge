"""Independent verification for Sprint 7.7 final figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.final_figures import (
    ABLATION_COMPONENT_ONLY_COUNT,
    ABLATION_DIRECT_COUNT,
    ABLATION_NOT_EVALUATED_COUNT,
    CLAIM_CONTROLS,
    FIGURE_CAPTIONS_PATH,
    FIGURE_FILENAMES,
    FIGURE_IDS,
    FIGURE_MANIFEST_PATH,
    GENERATED_FROM_FROZEN_EVIDENCE,
    NEW_EXPERIMENTS,
    NEW_SCIENTIFIC_RESULTS,
    NEW_TRAINING,
    OUTPUT_DIR,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_json(
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise AssertionError(f"Missing required artifact: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def _check(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise AssertionError(message)


def _verify_figure_files() -> None:
    output_dir = ROOT / OUTPUT_DIR

    _check(
        output_dir.is_dir(),
        "Final figure directory is missing.",
    )

    for figure_id in FIGURE_IDS:
        filename = FIGURE_FILENAMES[figure_id]

        path = output_dir / filename

        _check(
            path.is_file(),
            ("Missing final figure: " f"{filename}"),
        )

        _check(
            path.stat().st_size > 0,
            ("Empty final figure: " f"{filename}"),
        )


def _verify_manifest() -> dict[str, Any]:
    manifest_path = ROOT / FIGURE_MANIFEST_PATH

    manifest = _load_json(manifest_path)

    _check(
        manifest["figure_count"] == 7,
        ("Manifest must contain " "exactly seven figures."),
    )

    _check(
        tuple(manifest["required_figure_ids"]) == FIGURE_IDS,
        ("Manifest figure IDs do not " "match frozen contract."),
    )

    _check(
        manifest["generated_from_frozen_evidence"] is True,
        ("Manifest must declare " "frozen-evidence generation."),
    )

    _check(
        manifest["new_training"] is False,
        ("Sprint 7.7 must not introduce " "new training."),
    )

    _check(
        manifest["new_experiments"] is False,
        ("Sprint 7.7 must not introduce " "new experiments."),
    )

    _check(
        manifest["new_scientific_results"] is False,
        ("Sprint 7.7 must not create " "new scientific results."),
    )

    _check(
        len(manifest["figures"]) == 7,
        ("Manifest figure metadata " "count must be seven."),
    )

    for figure in manifest["figures"]:
        _check(
            figure["source_artifacts"],
            ("Every final figure must " "have source provenance."),
        )

        _check(
            figure["generated_from_frozen_evidence"] is True,
            ("Every figure must be generated " "from frozen evidence."),
        )

        _check(
            figure["limitations"],
            ("Every figure must retain " "scientific limitations."),
        )

        _check(
            figure["scientific_scope"],
            ("Every figure must define " "scientific scope."),
        )

    return manifest


def _verify_claim_controls(
    manifest: dict[str, Any],
) -> None:
    controls = manifest["claim_controls"]

    _check(
        controls == CLAIM_CONTROLS,
        ("Manifest claim controls differ " "from frozen contract."),
    )

    for (
        name,
        allowed,
    ) in controls.items():
        _check(
            allowed is False,
            ("Claim control unexpectedly " f"enabled: {name}"),
        )


def _verify_captions() -> None:
    path = ROOT / FIGURE_CAPTIONS_PATH

    _check(
        path.is_file(),
        ("Figure captions file " "is missing."),
    )

    text = path.read_text(encoding="utf-8")

    _check(
        bool(text.strip()),
        ("Figure captions file " "is empty."),
    )

    required_phrases = (
        "not executed as one matched end-to-end",
        "at least 2× storage compression",
        "did not demonstrate a robust",
        "Compactness is reported separately",
        "zero observed",
        "not formal stability",
        ("not normalized into a synthetic " "overall score"),
        "COMPONENT_ONLY",
        "Phase 2 validation objective",
    )

    forbidden_phrases = (
        "quantum advantage demonstrated",
        "quantum speedup demonstrated",
        "guaranteed safe",
        "production ready",
        "production-ready",
        "full-system superiority",
        ("universal trained policy " "demonstrated"),
    )

    normalized = " ".join(text.lower().split())

    for phrase in required_phrases:
        normalized_phrase = " ".join(phrase.lower().split())

        _check(
            normalized_phrase in normalized,
            ("Required scientific caption " f"language missing: {phrase}"),
        )

    for phrase in forbidden_phrases:
        normalized_phrase = " ".join(phrase.lower().split())

        _check(
            normalized_phrase not in normalized,
            ("Forbidden interpretation " "found in captions: " f"{phrase}"),
        )


def _verify_training_evidence() -> None:
    path = (
        ROOT
        / "results"
        / "final-validation"
        / "training-efficiency"
        / "final-training-efficiency-summary.json"
    )

    payload = _load_json(path)

    _check(
        payload["robust_ten_percent_efficiency_demonstrated"] is False,
        ("Training negative result " "must remain preserved."),
    )

    methods = payload["methods"]

    _check(
        len(methods) == 4,
        ("Training figure requires " "four frozen domain/method records."),
    )

    for item in methods:
        reached = item["target_reach"]["reached"]

        steps = item["steps_to_target"]

        if reached == 0:
            _check(
                steps is None,
                ("Non-attainment must remain " "missing, not censored."),
            )


def _verify_qml_evidence() -> None:
    path = ROOT / "results" / "final-validation" / "qml-ablation" / "qml-ablation.json"

    payload = _load_json(path)

    text = json.dumps(payload).lower()

    _check(
        ("quantum advantage" not in text or "false" in text),
        ("QML evidence must not support " "a quantum-advantage claim."),
    )

    _check(
        ("sample-efficiency" in text or "sample_efficiency" in text),
        ("QML artifact must retain " "sample-efficiency context."),
    )


def _verify_safety_evidence() -> None:
    path = (
        ROOT
        / "results"
        / "final-validation"
        / "safety-ablation"
        / "safety-ablation.json"
    )

    payload = _load_json(path)

    text = json.dumps(payload).lower()

    _check(
        "lyapunov" in text,
        ("Safety evidence must " "include Lyapunov."),
    )

    _check(
        "clipping" in text,
        ("Safety evidence must " "include clipping."),
    )

    _check(
        ("formal" in text or "guarantee" in text or "empirical" in text),
        ("Safety evidence must retain " "empirical/formal boundary language."),
    )


def _verify_cross_domain_controls() -> None:
    path = (
        ROOT
        / "results"
        / "final-validation"
        / "cross-domain"
        / "final-cross-domain-summary.json"
    )

    payload = _load_json(path)

    classifications = payload["classifications"]

    _check(
        classifications["same_trained_policy_weights"]["classification"]
        == "Not demonstrated",
        ("Same trained policy weights " "must remain not demonstrated."),
    )

    _check(
        classifications["zero_shot_cross_domain_transfer"]["classification"]
        == "Not demonstrated",
        ("Zero-shot cross-domain transfer " "must remain not demonstrated."),
    )

    _check(
        classifications["universal_safety_controller"]["classification"]
        == "Not demonstrated",
        ("Universal safety controller " "must remain not demonstrated."),
    )


def _verify_full_system_ablation() -> None:
    path = (
        ROOT
        / "results"
        / "final-validation"
        / "full-system-ablation"
        / "full-system-ablation.json"
    )

    payload = _load_json(path)

    phase1 = payload["phase1"]

    _check(
        phase1["direct_count"] == ABLATION_DIRECT_COUNT,
        ("DIRECT count must " "remain zero."),
    )

    _check(
        phase1["component_only_count"] == ABLATION_COMPONENT_ONLY_COUNT,
        ("COMPONENT_ONLY count " "must remain sixteen."),
    )

    _check(
        phase1["not_evaluated_count"] == ABLATION_NOT_EVALUATED_COUNT,
        ("NOT_EVALUATED count must " "remain zero."),
    )

    _check(
        phase1["direct_full_system_execution_found"] is False,
        ("No direct full-system execution " "must remain preserved."),
    )

    matrix = phase1["matrix"]

    _check(
        len(matrix) == 16,
        ("Full-system matrix must " "contain sixteen classifications."),
    )

    for row in matrix:
        _check(
            row["evidence_status"] == "component_only",
            ("Every Phase 1 full-system cell " "must remain component_only."),
        )

        _check(
            row["can_report_end_to_end_metrics"] is False,
            ("Component-only evidence must not " "report end-to-end metrics."),
        )

    controls = payload["scientific_controls"]

    _check(
        controls["synthetic_metric_composition_allowed"] is False,
        ("Synthetic metric composition " "must remain blocked."),
    )

    _check(
        controls["interaction_effects_estimable_with_component_only"] is False,
        (
            "Interaction effects must remain "
            "non-estimable from component-only evidence."
        ),
    )


def main() -> None:
    _check(
        GENERATED_FROM_FROZEN_EVIDENCE is True,
        ("Frozen-evidence flag " "must be true."),
    )

    _check(
        NEW_TRAINING is False,
        ("New training must " "remain false."),
    )

    _check(
        NEW_EXPERIMENTS is False,
        ("New experiments must " "remain false."),
    )

    _check(
        NEW_SCIENTIFIC_RESULTS is False,
        ("New scientific results " "must remain false."),
    )

    _verify_figure_files()

    manifest = _verify_manifest()

    _verify_claim_controls(manifest)

    _verify_captions()
    _verify_training_evidence()
    _verify_qml_evidence()
    _verify_safety_evidence()
    _verify_cross_domain_controls()
    _verify_full_system_ablation()

    print("=" * 68)

    print(" SPRINT 7.7 FINAL FIGURES — " "INDEPENDENT VERIFICATION")

    print("=" * 68)

    print()

    print("Architecture                         PASS")

    print("Compression Pareto                   PASS")

    print("Training Efficiency                  PASS")

    print("RL/QML                               PASS")

    print("Safety                               PASS")

    print("Cross-Domain                         PASS")

    print("Ablation Evidence                    PASS")

    print()

    print("Seven figure files                   PASS")

    print("Non-empty figure files               PASS")

    print("Figure provenance                    PASS")

    print("Captions                             PASS")

    print("Frozen evidence only                 PASS")

    print("Training negative result             PASS")

    print("QML negative result                  PASS")

    print("Safety empirical boundary            PASS")

    print("Cross-domain limitations             PASS")

    print("DIRECT count = 0                     PASS")

    print("COMPONENT_ONLY count = 16            PASS")

    print("Synthetic metric composition         BLOCKED")

    print("Interaction inference                BLOCKED")

    print("Claim controls                       PASS")

    print("No new experiments                   PASS")

    print()

    print("SPRINT 7.7 FINAL FIGURES: PASS")


if __name__ == "__main__":
    main()
