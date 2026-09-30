"""Independent verification for Sprint 7.8 final result tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.final_result_tables import (
    CLAIM_CONTROLS,
    FINAL_TABLES_JSON,
    FINAL_TABLES_MARKDOWN,
    NEW_EXPERIMENTS,
    NEW_SCIENTIFIC_RESULTS,
    NEW_TRAINING,
    NOT_APPLICABLE,
    NOT_MEASURED,
    NOT_REACHED,
    OUTPUT_DIR,
    SEEDS,
    STATISTICAL_PROTOCOL,
    SYNTHETIC_METRIC_COMPOSITION_ALLOWED,
    TABLE_FILENAMES,
    TABLE_IDS,
    TABLE_MANIFEST_PATH,
)

ROOT = Path(__file__).resolve().parents[1]


def _check(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise AssertionError(message)


def _load_json(
    path: Path,
) -> dict[str, Any]:
    _check(
        path.is_file(),
        f"Missing required JSON artifact: {path}",
    )

    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(
    path: Path,
) -> list[dict[str, str]]:
    _check(
        path.is_file(),
        f"Missing required CSV artifact: {path}",
    )

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def _table_path(
    table_id: str,
) -> Path:
    return ROOT / OUTPUT_DIR / TABLE_FILENAMES[table_id]


def _find_row(
    rows: list[dict[str, str]],
    **criteria: str,
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if all(row.get(key) == value for key, value in criteria.items())
    ]

    _check(
        len(matches) == 1,
        ("Expected exactly one matching row for " f"{criteria}, found {len(matches)}."),
    )

    return matches[0]


def _verify_output_package() -> None:
    output_dir = ROOT / OUTPUT_DIR

    _check(
        output_dir.is_dir(),
        "Sprint 7.8 table directory is missing.",
    )

    for table_id in TABLE_IDS:
        path = _table_path(table_id)

        _check(
            path.is_file(),
            ("Missing canonical CSV table: " f"{path.name}"),
        )

        _check(
            path.stat().st_size > 0,
            ("Canonical CSV table is empty: " f"{path.name}"),
        )

    for relative_path in (
        FINAL_TABLES_JSON,
        FINAL_TABLES_MARKDOWN,
        TABLE_MANIFEST_PATH,
    ):
        path = ROOT / relative_path

        _check(
            path.is_file(),
            ("Missing Sprint 7.8 package artifact: " f"{path}"),
        )

        _check(
            path.stat().st_size > 0,
            ("Empty Sprint 7.8 package artifact: " f"{path}"),
        )


def _verify_global_contract() -> None:
    _check(
        tuple(SEEDS)
        == (
            42,
            123,
            456,
        ),
        "Locked seed set changed.",
    )

    _check(
        "sample SD" in STATISTICAL_PROTOCOL,
        "Statistical protocol must retain sample SD.",
    )

    _check(
        NEW_TRAINING is False,
        "Sprint 7.8 must not introduce training.",
    )

    _check(
        NEW_EXPERIMENTS is False,
        "Sprint 7.8 must not introduce experiments.",
    )

    _check(
        NEW_SCIENTIFIC_RESULTS is False,
        ("Sprint 7.8 must not introduce " "new scientific results."),
    )

    _check(
        SYNTHETIC_METRIC_COMPOSITION_ALLOWED is False,
        ("Synthetic metric composition " "must remain blocked."),
    )

    _check(
        NOT_REACHED == "Not reached",
        "Not reached semantics changed.",
    )

    _check(
        NOT_MEASURED == "Not measured",
        "Not measured semantics changed.",
    )

    _check(
        NOT_APPLICABLE == "Not applicable",
        "Not applicable semantics changed.",
    )

    _check(
        len(
            {
                NOT_REACHED,
                NOT_MEASURED,
                NOT_APPLICABLE,
            }
        )
        == 3,
        ("Missing-value semantics must remain " "three distinct states."),
    )


def _verify_manifest() -> dict[str, Any]:
    manifest = _load_json(ROOT / TABLE_MANIFEST_PATH)

    _check(
        manifest["table_count"] == 5,
        ("Manifest must contain exactly " "five canonical tables."),
    )

    _check(
        tuple(manifest["required_table_ids"]) == TABLE_IDS,
        "Manifest table IDs differ from contract.",
    )

    _check(
        manifest["generated_from_frozen_evidence"] is True,
        ("Manifest must declare " "frozen-evidence generation."),
    )

    _check(
        manifest["new_training"] is False,
        "Manifest new_training must be false.",
    )

    _check(
        manifest["new_experiments"] is False,
        "Manifest new_experiments must be false.",
    )

    _check(
        manifest["new_scientific_results"] is False,
        ("Manifest new_scientific_results " "must be false."),
    )

    _check(
        manifest["synthetic_metric_composition"] is False,
        ("Manifest synthetic metric " "composition must be false."),
    )

    _check(
        manifest["claim_controls"] == CLAIM_CONTROLS,
        ("Manifest claim controls differ " "from frozen contract."),
    )

    _check(
        all(value is False for value in manifest["claim_controls"].values()),
        "All claim controls must remain blocked.",
    )

    table_entries = manifest["tables"]

    _check(
        len(table_entries) == 5,
        ("Manifest table metadata count " "must be five."),
    )

    for entry in table_entries:
        _check(
            entry["source_artifacts"],
            ("Every table must retain " "source provenance."),
        )

        _check(
            entry["limitations"],
            ("Every table must retain " "scientific limitations."),
        )

        _check(
            entry["statistical_protocol"] == STATISTICAL_PROTOCOL,
            ("Every table must retain the " "locked statistical protocol."),
        )

        _check(
            entry["claim_controls"] == CLAIM_CONTROLS,
            ("Every table must retain " "the claim controls."),
        )

    expected_counts = {
        "compression": 8,
        "training": 4,
        "rl": 6,
        "safety": 6,
        "cross_domain": 9,
    }

    for entry in table_entries:
        table_id = entry["table_id"]

        _check(
            entry["row_count"] == expected_counts[table_id],
            ("Unexpected row count for " f"{table_id}."),
        )

    return manifest


def _verify_final_json() -> dict[str, Any]:
    payload = _load_json(ROOT / FINAL_TABLES_JSON)

    _check(
        payload["status"] == "FROZEN",
        "Final result package must be FROZEN.",
    )

    _check(
        payload["generated_from_frozen_evidence"] is True,
        ("Final result package must use " "frozen evidence."),
    )

    _check(
        payload["new_training"] is False,
        "Final package new_training must be false.",
    )

    _check(
        payload["new_experiments"] is False,
        ("Final package new_experiments " "must be false."),
    )

    _check(
        payload["new_scientific_results"] is False,
        ("Final package new_scientific_results " "must be false."),
    )

    _check(
        payload["synthetic_metric_composition"] is False,
        ("Final package cannot contain " "synthetic metric composition."),
    )

    _check(
        payload["seeds"]
        == [
            42,
            123,
            456,
        ],
        "Final package seed set changed.",
    )

    _check(
        payload["missing_value_semantics"]
        == {
            "not_reached": "Not reached",
            "not_measured": "Not measured",
            "not_applicable": "Not applicable",
        },
        ("Final package missing-value " "semantics changed."),
    )

    _check(
        payload["claim_controls"] == CLAIM_CONTROLS,
        ("Final package claim controls " "differ from contract."),
    )

    _check(
        set(payload["tables"]) == set(TABLE_IDS),
        ("Final package must contain " "all five canonical tables."),
    )

    return payload


def _verify_compression() -> None:
    rows = _load_csv(_table_path("compression"))

    _check(
        len(rows) == 8,
        ("Compression table must contain " "eight rows."),
    )

    for domain in (
        "Driving",
        "Robotics",
    ):
        fp32 = _find_row(
            rows,
            Domain=domain,
            Method="FP32",
        )

        int8 = _find_row(
            rows,
            Domain=domain,
            Method="INT8",
        )

        svd = _find_row(
            rows,
            Domain=domain,
            Method="SVD",
        )

        tt_mps = _find_row(
            rows,
            Domain=domain,
            Method="TT/MPS",
        )

        _check(
            fp32["Criterion"] == "REFERENCE",
            (f"{domain} FP32 must remain " "the reference."),
        )

        _check(
            fp32["Relative ΔMSE"] == NOT_APPLICABLE,
            (f"{domain} FP32 ΔMSE must " "remain Not applicable."),
        )

        _check(
            int8["Criterion"] == "PASS",
            (f"{domain} INT8 PASS " "result changed."),
        )

        _check(
            svd["Criterion"] == "FAIL",
            (f"{domain} SVD FAIL " "result changed."),
        )

        _check(
            tt_mps["Criterion"] == "FAIL",
            (f"{domain} TT/MPS FAIL " "result changed."),
        )

        for row in (
            int8,
            svd,
            tt_mps,
        ):
            _check(
                "±" in row["MSE"],
                ("Seed-dependent compression MSE " "must preserve mean ± sample SD."),
            )

            _check(
                "±" in row["Relative ΔMSE"],
                ("Seed-dependent ΔMSE must preserve " "mean ± sample SD."),
            )

    for row in rows:
        _check(
            "±" not in row["Compression Ratio"],
            ("Deterministic compression ratio " "must not use fake ±0."),
        )


def _verify_training() -> None:
    rows = _load_csv(_table_path("training"))

    _check(
        len(rows) == 4,
        ("Training table must contain " "four rows."),
    )

    driving_svd = _find_row(
        rows,
        Domain="Driving",
        Method="Trainable SVD",
    )

    _check(
        driving_svd["Target Reached"] == "3/3",
        ("Driving SVD target reach " "must remain 3/3."),
    )

    _check(
        "282.667" in driving_svd["Steps to Target"],
        ("Driving SVD steps-to-target " "changed."),
    )

    _check(
        driving_svd["Efficiency Result"] == "FAIL",
        ("Driving SVD must not become " "a robust efficiency PASS."),
    )

    for row in rows:
        _check(
            row["Memory"] == NOT_MEASURED,
            ("Training memory was not measured " "and must remain explicit."),
        )

        _check(
            row["Efficiency Result"] == "FAIL",
            ("No training method may be presented " "as robust >=10% efficiency PASS."),
        )

        if row["Target Reached"] == "0/3":
            _check(
                row["Steps to Target"] == NOT_REACHED,
                ("Non-reaching training rows must " "remain Not reached."),
            )

            _check(
                row["Training Time"] == NOT_REACHED,
                ("Non-reaching training times must " "remain Not reached."),
            )


def _verify_rl() -> None:
    rows = _load_csv(_table_path("rl"))

    _check(
        len(rows) == 6,
        "RL/QML table must contain six rows.",
    )

    ppo_rows = [row for row in rows if row["Policy"] == "Classical PPO / MLP"]

    qml_rows = [row for row in rows if row["Policy"] == "QML / PQC"]

    matched_rows = [row for row in rows if row["Policy"] == "Matched classical control"]

    ppo_reaches = sum(int(row["Target Reaches"].split("/")[0]) for row in ppo_rows)

    qml_reaches = sum(int(row["Target Reaches"].split("/")[0]) for row in qml_rows)

    matched_reaches = sum(
        int(row["Target Reaches"].split("/")[0]) for row in matched_rows
    )

    _check(
        ppo_reaches == 6,
        ("Classical PPO target reaches " "must remain 6/6."),
    )

    _check(
        qml_reaches == 0,
        "QML target reaches must remain 0/6.",
    )

    _check(
        matched_reaches == 1,
        ("Matched-classical target reaches " "must remain 1/6."),
    )

    for row in qml_rows:
        _check(
            row["Steps to Target"] == NOT_REACHED,
            ("QML non-attainment must remain " "Not reached."),
        )

    matched_robotics = _find_row(
        rows,
        Domain="Robotics",
        Policy="Matched classical control",
    )

    _check(
        matched_robotics["Target Reaches"] == "1/3",
        ("Matched-classical robotics reach " "must remain 1/3."),
    )

    _check(
        matched_robotics["Steps to Target"] == "20000.000",
        ("The genuine matched-classical " "20,000-step reach changed."),
    )

    driving_qml = _find_row(
        rows,
        Domain="Driving",
        Policy="QML / PQC",
    )

    robotics_qml = _find_row(
        rows,
        Domain="Robotics",
        Policy="QML / PQC",
    )

    _check(
        driving_qml["Parameter Reduction"] == "95.90%",
        ("Driving PQC parameter reduction " "must remain 95.90%."),
    )

    _check(
        robotics_qml["Parameter Reduction"] == "95.51%",
        ("Robotics PQC parameter reduction " "must remain 95.51%."),
    )

    for row in rows:
        _check(
            "±" not in row["Actor Parameters"],
            ("Deterministic actor parameter count " "must not use fake ±0."),
        )


def _verify_safety() -> None:
    rows = _load_csv(_table_path("safety"))

    _check(
        len(rows) == 6,
        ("Safety table must contain " "six rows."),
    )

    for domain in (
        "Driving",
        "Robotics",
    ):
        for method in (
            "NONE",
            "CLIPPING",
            "LYAPUNOV",
        ):
            _find_row(
                rows,
                Domain=domain,
                Method=method,
            )

        clipping = _find_row(
            rows,
            Domain=domain,
            Method="CLIPPING",
        )

        lyapunov = _find_row(
            rows,
            Domain=domain,
            Method="LYAPUNOV",
        )

        _check(
            clipping["Violation Rate"].startswith("0.000"),
            (f"{domain} clipping must retain " "zero observed violation rate."),
        )

        _check(
            lyapunov["Violation Rate"].startswith("0.000"),
            (f"{domain} Lyapunov must retain " "zero observed violation rate."),
        )

        _check(
            lyapunov["Evidence Role"] == "Classical empirical safety filter",
            ("Lyapunov evidence must remain " "classical and empirical."),
        )


def _verify_cross_domain() -> None:
    rows = _load_csv(_table_path("cross_domain"))

    _check(
        len(rows) == 9,
        ("Cross-domain table must contain " "nine rows."),
    )

    architecture = _find_row(
        rows,
        **{"Component / Method": ("Shared AI/DL architecture")},
    )

    _check(
        "same trained policy weights: NO" in architecture["Shared Across Domains?"],
        ("Shared architecture must not become " "shared trained policy weights."),
    )

    full_system = _find_row(
        rows,
        **{"Component / Method": ("Full-system factorial")},
    )

    _check(
        full_system["Driving Result"] == "COMPONENT_ONLY",
        ("Driving full-system evidence must " "remain COMPONENT_ONLY."),
    )

    _check(
        full_system["Robotics Result"] == "COMPONENT_ONLY",
        ("Robotics full-system evidence must " "remain COMPONENT_ONLY."),
    )

    _check(
        "DIRECT = 0" in full_system["Evidence Boundary"],
        ("Full-system DIRECT count " "must remain zero."),
    )

    _check(
        "COMPONENT_ONLY = 16" in full_system["Evidence Boundary"],
        ("Full-system COMPONENT_ONLY count " "must remain sixteen."),
    )


def _verify_markdown() -> None:
    path = ROOT / FINAL_TABLES_MARKDOWN

    text = path.read_text(encoding="utf-8")

    _check(
        bool(text.strip()),
        "Reviewer Markdown is empty.",
    )

    mojibake_tokens = (
        "â€”",
        "Â±",
        "Ã—",
        "Î”",
    )

    for token in mojibake_tokens:
        _check(
            token not in text,
            ("UTF-8 mojibake is present in " f"the Markdown file: {token}"),
        )

    normalized = " ".join(text.lower().split())

    required_phrases = (
        "not reached",
        "not measured",
        "not applicable",
        "no quantum advantage",
        "quantum speedup",
        "qml sample-efficiency advantage",
        "zero observed violations",
        "not a formal stability proof",
        "no synthetic aggregate score",
        "same trained policy weights: no",
        "direct = 0",
        "component_only = 16",
        "tt/mps superiority: blocked",
        "full-system superiority: blocked",
        "production readiness / certification: blocked",
    )

    for phrase in required_phrases:
        _check(
            phrase in normalized,
            ("Required scientific boundary " f"missing from Markdown: {phrase}"),
        )

    _check(
        (
            "no quantum advantage, quantum speedup, "
            "or qml sample-efficiency advantage was demonstrated"
        )
        in normalized,
        ("The combined RL/QML negative-result statement " "is missing from Markdown."),
    )

    _check(
        "quantum advantage / speedup: blocked" in normalized,
        ("Quantum advantage / speedup claim control " "is missing from Markdown."),
    )

    _check(
        ("robust >=10% training-efficiency " "improvement: not demonstrated")
        in normalized,
        ("Training-efficiency negative conclusion " "is missing from Markdown."),
    )

    _check(
        "formal lyapunov stability: blocked" in normalized,
        ("Formal Lyapunov stability claim control " "is missing from Markdown."),
    )

    _check(
        "guaranteed zero violations: blocked" in normalized,
        ("Guaranteed-zero-violations claim control " "is missing from Markdown."),
    )

    _check(
        "zero-shot cross-domain transfer: blocked" in normalized,
        ("Zero-shot cross-domain transfer claim control " "is missing from Markdown."),
    )

    approved_negative_or_blocked_phrases = (
        (
            "no quantum advantage, quantum speedup, "
            "or qml sample-efficiency advantage was demonstrated"
        ),
        ("no qml sample-efficiency advantage " "or quantum advantage demonstrated"),
        (
            "phase 1 did not demonstrate a robust >=10% "
            "optimizer-step efficiency improvement across both "
            "domains and all locked seeds"
        ),
        "tt/mps superiority: blocked",
        ("robust >=10% training-efficiency " "improvement: not demonstrated"),
        "qml sample-efficiency advantage: blocked",
        "quantum advantage / speedup: blocked",
        "formal lyapunov stability: blocked",
        "guaranteed zero violations: blocked",
        "zero-shot cross-domain transfer: blocked",
        "full-system superiority: blocked",
        "production readiness / certification: blocked",
        "not a formal stability proof",
        "no formal stability guarantee",
        "no superiority claim",
        "no production runtime claim",
        "no synthetic aggregate score is permitted",
        "direct integrated evidence: no",
        "same trained policy weights: no",
        "same learned weights: no",
    )

    positive_claim_scan_text = normalized

    for phrase in approved_negative_or_blocked_phrases:
        positive_claim_scan_text = positive_claim_scan_text.replace(
            phrase,
            "",
        )

    forbidden_positive_claims = (
        "quantum advantage was demonstrated",
        "quantum advantage is demonstrated",
        "quantum advantage has been demonstrated",
        "quantum advantage achieved",
        "quantum advantage was achieved",
        "demonstrates quantum advantage",
        "demonstrated quantum advantage",
        "quantum speedup was demonstrated",
        "quantum speedup is demonstrated",
        "quantum speedup has been demonstrated",
        "quantum speedup achieved",
        "quantum speedup was achieved",
        "demonstrates quantum speedup",
        "demonstrated quantum speedup",
        "qml sample-efficiency advantage was demonstrated",
        "qml sample-efficiency advantage is demonstrated",
        "qml sample-efficiency advantage has been demonstrated",
        "qml sample-efficiency advantage achieved",
        "qml sample-efficiency advantage was achieved",
        "demonstrates qml sample-efficiency advantage",
        "demonstrated qml sample-efficiency advantage",
        "qml performance superiority was demonstrated",
        "qml performance superiority is demonstrated",
        "qml performance superiority achieved",
        "tt/mps superiority was demonstrated",
        "tt/mps superiority is demonstrated",
        "tt/mps superiority achieved",
        "tt/mps outperformed",
        "formal lyapunov stability was demonstrated",
        "formal lyapunov stability is demonstrated",
        "formal lyapunov stability was proven",
        "formal stability was proven",
        "formal stability guarantee",
        "guaranteed safe",
        "guaranteed safety",
        "production ready",
        "production-ready",
        "iso 26262 certified",
        "certified for production",
        "zero-shot cross-domain transfer was demonstrated",
        "zero-shot cross-domain transfer achieved",
        "zero-shot transfer was demonstrated",
        "zero-shot transfer achieved",
        "full-system superiority was demonstrated",
        "full-system superiority is demonstrated",
        "full-system superiority achieved",
        "shared trained policy weights: yes",
        "shared learned weights: yes",
        "direct integrated evidence: yes",
        "full-system direct evidence: yes",
    )

    for phrase in forbidden_positive_claims:
        _check(
            phrase not in positive_claim_scan_text,
            ("Forbidden positive interpretation found " f"in Markdown: {phrase}"),
        )

    _check(
        "safety potential" in normalized,
        ("Expected phrase 'safety potential' " "is missing or malformed."),
    )

    _check(
        "safetypotential" not in normalized,
        ("Malformed phrase 'safetypotential' " "found in Markdown."),
    )

    _check(
        "zero observed violations" in normalized,
        (
            "Safety evidence must explicitly use "
            "'zero observed violations' semantics."
        ),
    )

    _check(
        (
            "zero violation rate means zero observed violations "
            "under the frozen proxy evaluation only"
        )
        in normalized,
        ("Safety zero-violation limitation " "is missing from Markdown."),
    )

    _check(
        (
            "actor compactness is reported separately "
            "from sample efficiency and performance"
        )
        in normalized,
        ("QML compactness/performance separation " "is missing from Markdown."),
    )

    _check(
        (
            "no matched integrated compression x qml x safety "
            "factorial pipeline was directly executed in phase 1"
        )
        in normalized,
        ("Full-system component-only limitation " "is missing from Markdown."),
    )


def main() -> None:
    _verify_output_package()
    _verify_global_contract()
    _verify_manifest()
    _verify_final_json()
    _verify_compression()
    _verify_training()
    _verify_rl()
    _verify_safety()
    _verify_cross_domain()
    _verify_markdown()

    print("=" * 68)
    print(" SPRINT 7.8 FINAL RESULT TABLES — " "INDEPENDENT VERIFICATION")
    print("=" * 68)
    print()
    print("Compression table                  PASS")
    print("Training table                     PASS")
    print("RL/QML table                       PASS")
    print("Safety table                       PASS")
    print("Cross-domain table                 PASS")
    print()
    print("Statistics                         PASS")
    print("Missing-value semantics            PASS")
    print("Deterministic-value formatting     PASS")
    print("Provenance                         PASS")
    print("Claim controls                     PASS")
    print("No synthetic metrics               PASS")
    print("Phase 1 boundaries                 PASS")
    print("UTF-8 artifact integrity           PASS")
    print("Reviewer Markdown sanity           PASS")
    print()
    print("INT8 PASS both domains             PASS")
    print("SVD FAIL both domains              PASS")
    print("TT/MPS FAIL both domains           PASS")
    print("PPO target reaches = 6/6           PASS")
    print("QML target reaches = 0/6           PASS")
    print("Matched classical = 1/6            PASS")
    print("Full-system DIRECT = 0             PASS")
    print("Full-system COMPONENT_ONLY = 16    PASS")
    print()
    print("SPRINT 7.8 FINAL RESULT TABLES: PASS")


if __name__ == "__main__":
    main()
