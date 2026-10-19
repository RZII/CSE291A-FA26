import pytest

from agent.config import DEFAULT_PATH, load_config


def test_default_config_loads():
    cfg = load_config()
    assert cfg.agent.max_steps == 10
    assert cfg.llm.temperature == 0
    assert cfg.llm.enable_thinking is False


def test_misspelled_field_is_rejected(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text(DEFAULT_PATH.read_text().replace("max_tokens", "max_token"))
    with pytest.raises(TypeError, match="max_token"):
        load_config(bad)
