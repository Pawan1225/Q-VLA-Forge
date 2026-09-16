import json
from pathlib import Path

from q_vla_forge.evaluation.final_acceptance import (
    build_final_acceptance,
)
from q_vla_forge.evaluation.final_reproducibility import (
    sha256_file,
)


def write_batch_acceptance(
    final_root: Path,
    batch: str,
    passed: bool = True,
) -> None:
    path = final_root / f"batch-{batch}" / f"batch-{batch}-acceptance.json"

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            {
                "passed": passed,
            }
        ),
        encoding="utf-8",
    )


def write_reproducibility_result(
    final_root: Path,
    passed: bool = True,
) -> None:
    path = final_root / "reproducibility" / "final-reproducibility-verification.json"

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            {
                "passed": passed,
                "status": ("PASS" if passed else "FAIL"),
            }
        ),
        encoding="utf-8",
    )


def build_quality_gate(
    *,
    pytest_passed: bool = True,
    ruff_passed: bool = True,
    black_passed: bool = True,
) -> dict[str, dict[str, bool]]:
    return {
        "pytest": {
            "passed": pytest_passed,
        },
        "ruff": {
            "passed": ruff_passed,
        },
        "black": {
            "passed": black_passed,
        },
    }


def test_sha256_is_deterministic(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.txt"

    path.write_text(
        "q-vla-forge",
        encoding="utf-8",
    )

    first = sha256_file(path)

    second = sha256_file(path)

    assert first == second
    assert len(first) == 64


def test_final_acceptance_passes(
    tmp_path: Path,
) -> None:
    final_root = tmp_path / "results" / "final-validation"

    for batch in (
        "a",
        "b",
        "c",
        "d",
    ):
        write_batch_acceptance(
            final_root,
            batch,
        )

    write_reproducibility_result(final_root)

    quality_gate = build_quality_gate()

    result = build_final_acceptance(
        tmp_path,
        quality_gate,
    )

    assert result["passed"]
    assert result["sprint7_complete"]
    assert result["phase1_evidence_frozen"]
    assert result["ready_for_submission_packaging"]
    assert result["status"] == "PASS"


def test_final_acceptance_fails_quality_gate(
    tmp_path: Path,
) -> None:
    final_root = tmp_path / "results" / "final-validation"

    for batch in (
        "a",
        "b",
        "c",
        "d",
    ):
        write_batch_acceptance(
            final_root,
            batch,
        )

    write_reproducibility_result(final_root)

    quality_gate = build_quality_gate(
        ruff_passed=False,
    )

    result = build_final_acceptance(
        tmp_path,
        quality_gate,
    )

    assert not result["passed"]
    assert not result["sprint7_complete"]
    assert not result["phase1_evidence_frozen"]
    assert not result["ready_for_submission_packaging"]
    assert result["status"] == "FAIL"


def test_final_acceptance_fails_batch(
    tmp_path: Path,
) -> None:
    final_root = tmp_path / "results" / "final-validation"

    for batch in (
        "a",
        "b",
        "c",
        "d",
    ):
        write_batch_acceptance(
            final_root,
            batch,
            passed=(batch != "c"),
        )

    write_reproducibility_result(final_root)

    quality_gate = build_quality_gate()

    result = build_final_acceptance(
        tmp_path,
        quality_gate,
    )

    assert not result["passed"]
    assert not result["sprint7_complete"]
    assert result["status"] == "FAIL"


def test_final_acceptance_fails_reproducibility(
    tmp_path: Path,
) -> None:
    final_root = tmp_path / "results" / "final-validation"

    for batch in (
        "a",
        "b",
        "c",
        "d",
    ):
        write_batch_acceptance(
            final_root,
            batch,
        )

    write_reproducibility_result(
        final_root,
        passed=False,
    )

    quality_gate = build_quality_gate()

    result = build_final_acceptance(
        tmp_path,
        quality_gate,
    )

    assert not result["passed"]
    assert not result["sprint7_complete"]
    assert result["status"] == "FAIL"
