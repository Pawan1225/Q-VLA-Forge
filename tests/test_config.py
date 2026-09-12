from pathlib import Path

import pytest

from q_vla_forge.utils.config import load_yaml_config


def test_load_yaml_config(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "experiment:\n  name: test\nseed: 42\n",
        encoding="utf-8",
    )

    config = load_yaml_config(config_file)

    assert config["experiment"]["name"] == "test"
    assert config["seed"] == 42


def test_empty_yaml_returns_empty_dict(tmp_path: Path) -> None:
    config_file = tmp_path / "empty.yaml"
    config_file.write_text("", encoding="utf-8")

    assert load_yaml_config(config_file) == {}


def test_missing_config_raises_error(tmp_path: Path) -> None:
    missing_file = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError):
        load_yaml_config(missing_file)


def test_non_mapping_yaml_raises_error(tmp_path: Path) -> None:
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text(
        "- item1\n- item2\n",
        encoding="utf-8",
    )

    with pytest.raises(TypeError):
        load_yaml_config(config_file)
