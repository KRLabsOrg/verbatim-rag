"""Tests for the file checks of TemplateManager.load (#124)."""

import json
import logging

import pytest
from verbatim_core.templates.manager import TemplateManager


@pytest.mark.parametrize(
    "state",
    [
        pytest.param({"current_mode": "static", "citation_mode": "sometimes"}, id="bad_citation_mode"),
        pytest.param({"current_mode": "static", "citation_format": "{unknown}"}, id="bad_citation_format"),
        pytest.param(["current_mode", "static"], id="top_level_list"),
        pytest.param({"current_mode": "static", "strategies": ["static"]}, id="strategies_list"),
        pytest.param({"current_mode": ["static"]}, id="mode_is_not_a_name"),
        pytest.param({"current_mode": "static", "strategies": {"static": ["template"]}}, id="state_is_not_an_object"),
        pytest.param({"current_mode": "random", "citation_mode": "sometimes"}, id="mode_before_a_bad_setting"),
    ],
)
def test_a_file_the_manager_cannot_apply_leaves_it_unchanged(tmp_path, state, caplog, capsys):
    (tmp_path / "templates.json").write_text(json.dumps(state))
    manager = TemplateManager(llm_client=None)
    config = manager.config.model_dump()
    states = {mode: strategy.save_state() for mode, strategy in manager.strategies.items() if strategy}

    assert manager.load(str(tmp_path / "templates.json")) is False

    assert manager.config.model_dump() == config
    assert {mode: strategy.save_state() for mode, strategy in manager.strategies.items() if strategy} == states
    assert [record.levelno for record in caplog.records] == [logging.WARNING]
    assert capsys.readouterr() == ("", "")


def test_a_valid_file_still_loads(tmp_path):
    state = {"current_mode": "random", "strategies": {"random": {"templates": ["One: [DISPLAY_SPANS]"]}}}
    (tmp_path / "templates.json").write_text(json.dumps(state))
    manager = TemplateManager(llm_client=None)

    assert manager.load(str(tmp_path / "templates.json")) is True

    assert manager.current_mode == "random"
    assert manager.strategies["random"].templates == ["One: [DISPLAY_SPANS]"]