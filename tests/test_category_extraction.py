"""Tests for spec 8 (categories-from-prompt): the standalone category-extraction
tool that turns a free-text instruction file into a ``categories.json``, via one
``Classifier.classify()`` invocation. No live network/provider calls — every LLM
call is faked, per ``AGENTS.md``'s testing policy.

Direct imports of ``query_classification.category_extraction`` (all functions),
``schema.build_category_extraction_model``, and
``prompts.build_category_extraction_prompt`` are an accepted INV-7 exception —
see ``spec/ARCHITECTURE.md`` and this spec's own ``ADR.md``.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from query_classification.schema import build_category_extraction_model


# ---------------------------------------------------------------------------
# Task 1 / AR-1.2: schema.build_category_extraction_model()
# ---------------------------------------------------------------------------


def test_build_category_extraction_model_strict_mode_shape():
    from litellm.utils import type_to_response_format_param

    model = build_category_extraction_model()
    schema = type_to_response_format_param(model)
    assert schema["json_schema"]["strict"] is True
    labels_field = schema["json_schema"]["schema"]["properties"]["labels"]
    assert "minItems" not in labels_field
    assert "maxItems" not in labels_field


def test_build_category_extraction_model_extra_forbid_on_outer_model():
    model = build_category_extraction_model()
    assert model.model_config.get("extra") == "forbid"


def test_build_category_extraction_model_exactly_two_fields():
    model = build_category_extraction_model()
    assert set(model.model_fields.keys()) == {"category_description", "labels"}


def test_build_category_extraction_model_accepts_well_formed_payload():
    model = build_category_extraction_model()
    raw = json.dumps(
        {
            "category_description": "d",
            "labels": [{"value": "billing", "description": "d"}],
        }
    )
    result = model.model_validate_json(raw)
    assert result.category_description == "d"
    assert result.labels[0].value == "billing"


def test_build_category_extraction_model_rejects_extra_top_level_field():
    model = build_category_extraction_model()
    raw = json.dumps(
        {
            "category_description": "d",
            "labels": [{"value": "billing", "description": "d"}],
            "bogus": "y",
        }
    )
    with pytest.raises(ValidationError):
        model.model_validate_json(raw)


def test_build_category_extraction_model_rejects_extra_nested_label_field():
    model = build_category_extraction_model()
    raw = json.dumps(
        {
            "category_description": "d",
            "labels": [{"value": "billing", "description": "d", "extra": "x"}],
        }
    )
    with pytest.raises(ValidationError):
        model.model_validate_json(raw)


# ---------------------------------------------------------------------------
# Task 2 / AR-1.3: resources.py + prompts.build_category_extraction_prompt()
# ---------------------------------------------------------------------------


def test_default_category_extraction_prompt_file_exists():
    from query_classification.resources import DEFAULT_CATEGORY_EXTRACTION_PROMPT_FILE

    assert DEFAULT_CATEGORY_EXTRACTION_PROMPT_FILE.exists()


def test_build_category_extraction_prompt_returns_bundled_file_contents():
    from query_classification.prompts import build_category_extraction_prompt
    from query_classification.resources import DEFAULT_CATEGORY_EXTRACTION_PROMPT_FILE

    result = build_category_extraction_prompt()
    assert result == DEFAULT_CATEGORY_EXTRACTION_PROMPT_FILE.read_text()


def test_build_category_extraction_prompt_override_file(tmp_path):
    from query_classification.prompts import build_category_extraction_prompt

    custom = tmp_path / "custom.txt"
    custom.write_text("a custom extraction prompt")
    result = build_category_extraction_prompt(custom)
    assert result == "a custom extraction prompt"


# ---------------------------------------------------------------------------
# Task 3 / FR-1.1, FR-1.3, FR-1.4, FR-1.5: extract_category_from_prompt()
# ---------------------------------------------------------------------------

from query_classification.category_extraction import (  # noqa: E402
    _check_reserved_sentinel,
    extract_category_from_prompt,
)


class _FakeClassifier:
    """Duck-typed fake -- extract_category_from_prompt only calls .classify(text)."""

    def __init__(self, response):
        self.response = response
        self.calls = []

    def classify(self, text):
        self.calls.append(text)
        return self.response


def test_extract_category_round_trips_three_labels_verbatim():
    response = {
        "category_description": "ticket topic",
        "labels": [
            {"value": "billing", "description": "billing-related"},
            {"value": "technical", "description": "technical issue"},
            {"value": "account", "description": "account management"},
        ],
    }
    fake = _FakeClassifier(response)
    category = extract_category_from_prompt("instructions", "topic", fake)
    assert category.name == "topic"
    assert category.description == "ticket topic"
    assert [(lbl.value, lbl.description) for lbl in category.labels] == [
        ("billing", "billing-related"),
        ("technical", "technical issue"),
        ("account", "account management"),
    ]
    assert fake.calls == ["instructions"]


def test_extract_category_zero_labels_raises():
    fake = _FakeClassifier({"category_description": "d", "labels": []})
    with pytest.raises(ValueError, match="at least one label"):
        extract_category_from_prompt("instructions", "topic", fake)


def test_extract_category_duplicate_label_values_raise():
    response = {
        "category_description": "d",
        "labels": [
            {"value": "urgent", "description": "d1"},
            {"value": "urgent", "description": "d2"},
        ],
    }
    fake = _FakeClassifier(response)
    with pytest.raises(ValueError, match="urgent"):
        extract_category_from_prompt("instructions", "topic", fake)


@pytest.mark.parametrize("reserved_value", ["none", "NONE", "none - invented"])
def test_extract_category_reserved_sentinel_values_raise(reserved_value):
    response = {
        "category_description": "d",
        "labels": [{"value": reserved_value, "description": "d"}],
    }
    fake = _FakeClassifier(response)
    with pytest.raises(ValueError, match="reserved"):
        extract_category_from_prompt("instructions", "topic", fake)


def test_extract_category_blank_category_description_raises():
    response = {"category_description": "   ", "labels": [{"value": "a", "description": "d"}]}
    fake = _FakeClassifier(response)
    with pytest.raises(ValueError):
        extract_category_from_prompt("instructions", "topic", fake)


def test_extract_category_blank_label_value_raises():
    response = {"category_description": "d", "labels": [{"value": "", "description": "d"}]}
    fake = _FakeClassifier(response)
    with pytest.raises(ValueError):
        extract_category_from_prompt("instructions", "topic", fake)


def test_extract_category_blank_label_description_raises():
    response = {"category_description": "d", "labels": [{"value": "a", "description": "   "}]}
    fake = _FakeClassifier(response)
    with pytest.raises(ValueError):
        extract_category_from_prompt("instructions", "topic", fake)


@pytest.mark.parametrize(
    "value,expect_reject",
    [
        ("none", True),
        ("NONE", True),
        (" None ", True),
        ("none - invented", True),
        ("none-invented", False),
        ("nonexistent", False),
        ("positive", False),
    ],
)
def test_check_reserved_sentinel_matches_experiment_module(value, expect_reject):
    """Parity check: the duplicated _check_reserved_sentinel must behave
    identically to experiment.py's own function for the same inputs -- not
    just 'looks similar'."""
    from query_classification.experiment import _check_reserved_sentinel as experiment_check

    def _raises(fn):
        try:
            fn([value])
            return False
        except ValueError:
            return True

    assert _raises(lambda vs: _check_reserved_sentinel(vs)) == expect_reject
    assert _raises(lambda vs: experiment_check(vs)) == expect_reject


# ---------------------------------------------------------------------------
# Task 4 / FR-1.2, FR-1.6, FR-1.7, FR-1.8, AR-1.3, AR-1.4, AR-1.7: CLI
# ---------------------------------------------------------------------------

import sys as _sys  # noqa: E402

import litellm  # noqa: E402

from query_classification import category_extraction as category_extraction_module  # noqa: E402
from query_classification.categories import load_categories  # noqa: E402


class _FakeUsage:
    def __init__(self, prompt_tokens=10, completion_tokens=5):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


def _fake_completion(calls, labels=None):
    if labels is None:
        labels = [
            {"value": "billing", "description": "billing issues"},
            {"value": "technical", "description": "technical issues"},
        ]

    def fake_completion(**kwargs):
        calls.append(kwargs)
        payload = json.dumps({"category_description": "ticket topic", "labels": labels})

        class _Msg:
            content = payload

        class _Choice:
            message = _Msg()
            finish_reason = "stop"

        class _Resp:
            choices = [_Choice()]
            usage = _FakeUsage()

        return _Resp()

    return fake_completion


@pytest.fixture
def fake_completion(monkeypatch):
    """Fakes litellm.completion (not Classifier.classify) so main()'s own
    Classifier construction/wiring is exercised for real, per the plan's own
    requirement. Also neutralizes load_dotenv (this repo's real .env sets
    DEFAULT_LLM_PROVIDER=cerebus, picked up merely by importing litellm) and
    the AWS-backed Cerebus key resolver, so these tests stay offline and
    deterministic regardless of local .env/AWS-SSO state."""
    import query_classification.classifier as classifier_module

    monkeypatch.setattr(category_extraction_module, "load_dotenv", lambda *a, **k: None)
    # Importing litellm itself triggers a load_dotenv() that picks up this
    # repo's real .env (DEFAULT_LLM_PROVIDER=cerebus) into the process
    # environment before any test runs -- neutralizing this module's own
    # load_dotenv call isn't sufficient to undo that already-polluted env var.
    monkeypatch.delenv("DEFAULT_LLM_PROVIDER", raising=False)
    monkeypatch.setattr(classifier_module, "_resolve_cerebus_api_key", lambda: "test-key")
    calls = []
    monkeypatch.setattr(litellm, "completion", _fake_completion(calls))
    return calls


def _run_main(argv):
    _sys.argv = ["extract_categories.py"] + argv
    try:
        category_extraction_module.main()
        return 0
    except SystemExit as e:
        return e.code


def test_build_parser_requires_prompt_file_and_category_name():
    with pytest.raises(SystemExit):
        category_extraction_module.build_parser().parse_args([])


@pytest.mark.parametrize("bad_name", ["__root__", "model_config", "not an identifier"])
def test_cli_rejects_invalid_category_name_before_any_llm_call(tmp_path, fake_completion, bad_name):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("classify tickets by topic")
    code = _run_main(
        ["--prompt-file", str(prompt_file), "--category-name", bad_name,
         "--output", str(tmp_path / "out.json")]
    )
    assert code == 1
    assert len(fake_completion) == 0


def test_cli_successful_run_produces_valid_category_name(tmp_path, fake_completion):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("classify tickets by topic")
    output_path = tmp_path / "out.json"
    code = _run_main(
        ["--prompt-file", str(prompt_file), "--category-name", "topic",
         "--output", str(output_path)]
    )
    assert code in (0, None)
    categories = load_categories(output_path)
    assert len(categories) == 1
    assert categories[0].name == "topic"
    assert len(fake_completion) == 1


def test_cli_zero_labels_fails_and_writes_no_output(tmp_path, monkeypatch):
    import query_classification.classifier as classifier_module

    monkeypatch.setattr(category_extraction_module, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("DEFAULT_LLM_PROVIDER", raising=False)
    monkeypatch.setattr(classifier_module, "_resolve_cerebus_api_key", lambda: "test-key")
    calls = []
    monkeypatch.setattr(litellm, "completion", _fake_completion(calls, labels=[]))

    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("classify tickets by topic")
    output_path = tmp_path / "out.json"
    code = _run_main(
        ["--prompt-file", str(prompt_file), "--category-name", "topic",
         "--output", str(output_path)]
    )
    assert code == 1
    assert not output_path.exists()


def test_cli_output_already_exists_without_overwrite_fails_second_run(tmp_path, fake_completion):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("classify tickets by topic")
    output_path = tmp_path / "out.json"

    code = _run_main(
        ["--prompt-file", str(prompt_file), "--category-name", "topic",
         "--output", str(output_path)]
    )
    assert code in (0, None)
    assert len(fake_completion) == 1
    first_content = output_path.read_text()

    code = _run_main(
        ["--prompt-file", str(prompt_file), "--category-name", "topic",
         "--output", str(output_path)]
    )
    assert code == 1
    assert len(fake_completion) == 1  # no further LLM call

    code = _run_main(
        ["--prompt-file", str(prompt_file), "--category-name", "topic2",
         "--output", str(output_path), "--overwrite"]
    )
    assert code in (0, None)
    assert len(fake_completion) == 2
    assert output_path.read_text() != first_content
    assert load_categories(output_path)[0].name == "topic2"


def test_cli_symlinked_output_fails_before_any_llm_call(tmp_path, fake_completion):
    real_target = tmp_path / "real.json"
    real_target.write_text('{"categories": []}')
    symlink_path = tmp_path / "out.json"
    symlink_path.symlink_to(real_target)

    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("classify tickets by topic")

    for extra in ([], ["--overwrite"]):
        code = _run_main(
            ["--prompt-file", str(prompt_file), "--category-name", "topic",
             "--output", str(symlink_path)] + extra
        )
        assert code == 1
    assert len(fake_completion) == 0


def test_cli_output_is_existing_directory_fails_before_any_llm_call(tmp_path, fake_completion):
    output_dir = tmp_path / "out.json"
    output_dir.mkdir()
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("classify tickets by topic")
    code = _run_main(
        ["--prompt-file", str(prompt_file), "--category-name", "topic",
         "--output", str(output_dir)]
    )
    assert code == 1
    assert len(fake_completion) == 0


def test_cli_output_missing_parent_directory_fails_before_any_llm_call(tmp_path, fake_completion):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("classify tickets by topic")
    code = _run_main(
        ["--prompt-file", str(prompt_file), "--category-name", "topic",
         "--output", str(tmp_path / "missing_dir" / "out.json")]
    )
    assert code == 1
    assert len(fake_completion) == 0


@pytest.mark.parametrize("content", ["", "   \n  "])
def test_cli_empty_or_whitespace_prompt_file_fails_before_any_llm_call(tmp_path, fake_completion, content):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text(content)
    code = _run_main(
        ["--prompt-file", str(prompt_file), "--category-name", "topic",
         "--output", str(tmp_path / "out.json")]
    )
    assert code == 1
    assert len(fake_completion) == 0


def test_cli_main_constructs_classifier_correctly(tmp_path, monkeypatch, fake_completion):
    from query_classification.classifier import Classifier

    recorded = {}
    original_init = Classifier.__init__

    def _spy_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        recorded["model_id"] = self.model_id
        recorded["system_prompt"] = self.system_prompt
        recorded["max_retries"] = self.max_retries
        recorded["api_key"] = self.api_key
        recorded["extra_headers"] = self.extra_headers

    monkeypatch.setattr(Classifier, "__init__", _spy_init)

    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("classify tickets by topic")
    code = _run_main(
        ["--prompt-file", str(prompt_file), "--category-name", "topic",
         "--output", str(tmp_path / "out.json"), "--model", "gpt-4o-mini", "--retries", "5"]
    )
    assert code in (0, None)
    assert recorded["model_id"] == "gpt-4o-mini"
    assert recorded["max_retries"] == 5
    assert "extract" in recorded["system_prompt"].lower() or len(recorded["system_prompt"]) > 0
    assert recorded["api_key"] is None
    assert recorded["extra_headers"] is None


def test_cli_cerebus_rejects_insecure_api_base_before_any_llm_call(tmp_path, fake_completion):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("classify tickets by topic")
    code = _run_main(
        ["--prompt-file", str(prompt_file), "--category-name", "topic",
         "--output", str(tmp_path / "out.json"), "--cerebus",
         "--api-base", "http://insecure.example/v1"]
    )
    assert code == 1
    assert len(fake_completion) == 0


def test_cli_handled_failure_prints_error_no_traceback_exit_1(tmp_path, fake_completion, capsys):
    code = _run_main(
        ["--prompt-file", str(tmp_path / "does_not_exist.txt"), "--category-name", "topic",
         "--output", str(tmp_path / "out.json")]
    )
    assert code == 1
    out = capsys.readouterr().out
    assert out.startswith("Error: ")
    assert "Traceback" not in out


def test_cli_success_prints_confirmation_with_output_path_and_exits_zero(tmp_path, fake_completion, capsys):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("classify tickets by topic")
    output_path = tmp_path / "out.json"
    code = _run_main(
        ["--prompt-file", str(prompt_file), "--category-name", "topic",
         "--output", str(output_path)]
    )
    assert code in (0, None)
    out = capsys.readouterr().out
    assert str(output_path) in out


def test_cli_missing_bundled_resource_names_extraction_prompt_flag(tmp_path, monkeypatch, fake_completion, capsys):
    import query_classification.resources as resources_module

    monkeypatch.setattr(resources_module, "RESOURCES_DIR", tmp_path / "nonexistent")
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("classify tickets by topic")
    code = _run_main(
        ["--prompt-file", str(prompt_file), "--category-name", "topic",
         "--output", str(tmp_path / "out.json")]
    )
    assert code == 1
    out = capsys.readouterr().out
    assert "--extraction-prompt" in out
    assert "--categories" not in out
    assert "--task-description" not in out
