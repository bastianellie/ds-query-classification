"""Offline tests for the Cerebus/Portkey gateway support in classifier.py.

No live network, AWS, or provider calls — AWS Secrets Manager is exercised
via a fake ``boto3`` module injected into ``sys.modules``.
"""

from __future__ import annotations

import sys
import types

import pytest

from query_classification import Classifier
from query_classification import classifier as classifier_module
from query_classification.classifier import (
    build_cerebus_completion_kwargs,
    cerebus_enabled_via_env,
    cerebus_model_id,
)


@pytest.fixture(autouse=True)
def _reset_cerebus_cache():
    """The resolved Cerebus key is cached at module level (in-process only,
    never persisted to disk) — reset it so tests don't leak state."""
    classifier_module._cerebus_api_key_cache = None
    yield
    classifier_module._cerebus_api_key_cache = None


@pytest.fixture
def fake_boto3(monkeypatch):
    """A minimal fake `boto3` module whose Secrets Manager client returns a
    configurable secret payload, injected via sys.modules like the other
    optional-dependency fakes in this test suite."""
    calls = {
        "session_count": 0,
        "get_secret_value_count": 0,
        "profile_name": None,
        "region_name": None,
        "secret_id": None,
    }
    state = {"secret_string": '{"sciencedirect_portkey_api_key": "fetched-key-abc"}', "raise_on_call": None}

    class FakeSecretsManagerClient:
        def get_secret_value(self, SecretId):
            calls["get_secret_value_count"] += 1
            calls["secret_id"] = SecretId
            if state["raise_on_call"] is not None:
                raise state["raise_on_call"]
            return {"SecretString": state["secret_string"]}

    class FakeSession:
        def __init__(self, profile_name=None):
            calls["session_count"] += 1
            calls["profile_name"] = profile_name
            self.profile_name = profile_name

        def client(self, service_name, region_name=None):
            assert service_name == "secretsmanager"
            calls["region_name"] = region_name
            return FakeSecretsManagerClient()

    fake_mod = types.ModuleType("boto3")
    fake_mod.Session = FakeSession
    monkeypatch.setitem(sys.modules, "boto3", fake_mod)
    return calls, state


# ---------------------------------------------------------------------------
# _resolve_cerebus_api_key
# ---------------------------------------------------------------------------


def test_resolve_key_from_env_var_skips_aws(monkeypatch, fake_boto3):
    calls, _ = fake_boto3
    monkeypatch.setenv("CEREBUS_API_KEY", "explicit-key")
    key = classifier_module._resolve_cerebus_api_key()
    assert key == "explicit-key"
    assert calls["session_count"] == 0  # never touched AWS


def test_resolve_key_falls_back_to_aws_secrets_manager(monkeypatch, fake_boto3):
    calls, _ = fake_boto3
    monkeypatch.delenv("CEREBUS_API_KEY", raising=False)
    key = classifier_module._resolve_cerebus_api_key()
    assert key == "fetched-key-abc"
    assert calls["session_count"] == 1


def test_resolve_key_uses_custom_secret_coordinates(monkeypatch, fake_boto3):
    calls, state = fake_boto3
    state["secret_string"] = '{"my_custom_key": "custom-value"}'
    monkeypatch.delenv("CEREBUS_API_KEY", raising=False)
    monkeypatch.setenv("CEREBUS_SECRET_KEY", "my_custom_key")
    key = classifier_module._resolve_cerebus_api_key()
    assert key == "custom-value"


def test_resolve_key_uses_documented_default_aws_coordinates(monkeypatch, fake_boto3):
    calls, _ = fake_boto3
    monkeypatch.delenv("CEREBUS_API_KEY", raising=False)
    for var in ("CEREBUS_AWS_PROFILE", "CEREBUS_AWS_REGION", "CEREBUS_SECRET_ID", "CEREBUS_SECRET_KEY"):
        monkeypatch.delenv(var, raising=False)
    classifier_module._resolve_cerebus_api_key()
    assert calls["profile_name"] == "kd-nonprod"
    assert calls["region_name"] == "us-east-1"
    assert calls["secret_id"] == "shared_genai/portkey-nonprod"


def test_resolve_key_passes_through_overridden_aws_coordinates(monkeypatch, fake_boto3):
    calls, _ = fake_boto3
    monkeypatch.delenv("CEREBUS_API_KEY", raising=False)
    monkeypatch.setenv("CEREBUS_AWS_PROFILE", "my-profile")
    monkeypatch.setenv("CEREBUS_AWS_REGION", "eu-west-1")
    monkeypatch.setenv("CEREBUS_SECRET_ID", "my/secret")
    classifier_module._resolve_cerebus_api_key()
    assert calls["profile_name"] == "my-profile"
    assert calls["region_name"] == "eu-west-1"
    assert calls["secret_id"] == "my/secret"


def test_resolve_key_caches_after_first_aws_fetch(monkeypatch, fake_boto3):
    calls, _ = fake_boto3
    monkeypatch.delenv("CEREBUS_API_KEY", raising=False)
    classifier_module._resolve_cerebus_api_key()
    classifier_module._resolve_cerebus_api_key()
    assert calls["session_count"] == 1  # second call reused the cache


def test_resolve_key_missing_boto3_raises_actionable_error(monkeypatch):
    monkeypatch.delenv("CEREBUS_API_KEY", raising=False)
    monkeypatch.delitem(sys.modules, "boto3", raising=False)
    monkeypatch.setattr(
        "builtins.__import__",
        _raise_module_not_found_for_boto3(__import__),
    )
    with pytest.raises(RuntimeError, match=r"pip install '\.\[cerebus\]'"):
        classifier_module._resolve_cerebus_api_key()


def test_resolve_key_missing_transitive_dependency_is_not_misreported_as_boto3(monkeypatch):
    """A ModuleNotFoundError for something OTHER than boto3 itself (e.g. one
    of boto3's own dependencies) must propagate as-is, not get relabeled as
    'boto3 is not installed' — that would send someone in circles reinstalling
    a package that's already present."""
    monkeypatch.delenv("CEREBUS_API_KEY", raising=False)
    monkeypatch.delitem(sys.modules, "boto3", raising=False)
    real_import = __import__

    def _fake_import(name, *args, **kwargs):
        if name == "boto3":
            raise ModuleNotFoundError("No module named 'some_boto3_dependency'", name="some_boto3_dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _fake_import)
    with pytest.raises(ModuleNotFoundError, match="some_boto3_dependency"):
        classifier_module._resolve_cerebus_api_key()


def _raise_module_not_found_for_boto3(real_import):
    def _fake_import(name, *args, **kwargs):
        if name == "boto3":
            # Real import failures always set .name — match that exactly,
            # since the code under test branches on e.name == "boto3" (to
            # distinguish "boto3 itself is missing" from "one of boto3's own
            # transitive dependencies is missing").
            raise ModuleNotFoundError("No module named 'boto3'", name="boto3")
        return real_import(name, *args, **kwargs)

    return _fake_import


def test_resolve_key_aws_failure_raises_clear_error_with_remediation(monkeypatch, fake_boto3):
    _, state = fake_boto3
    state["raise_on_call"] = RuntimeError("access denied")
    monkeypatch.delenv("CEREBUS_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="aws sso login"):
        classifier_module._resolve_cerebus_api_key()


def test_resolve_key_missing_json_key_raises(monkeypatch, fake_boto3):
    _, state = fake_boto3
    state["secret_string"] = '{"some_other_key": "x"}'
    monkeypatch.delenv("CEREBUS_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        classifier_module._resolve_cerebus_api_key()


# ---------------------------------------------------------------------------
# build_cerebus_completion_kwargs
# ---------------------------------------------------------------------------


def test_build_kwargs_mode_defaults_to_direct(monkeypatch):
    """CEREBUS_MODE is optional: unset means 'direct', not an error, so the
    minimal-footprint setup (DEFAULT_LLM_PROVIDER=cerebus + optional
    CEREBUS_API_KEY, nothing else) works out of the box."""
    monkeypatch.setenv("CEREBUS_API_KEY", "k")
    monkeypatch.delenv("CEREBUS_MODE", raising=False)
    monkeypatch.delenv("CEREBUS_GATEWAY_DIRECT_URL", raising=False)
    result = build_cerebus_completion_kwargs()
    assert result["extra_headers"]["x-portkey-provider"] == "openai"


def test_build_kwargs_rejects_invalid_mode(monkeypatch):
    monkeypatch.setenv("CEREBUS_MODE", "not-a-real-mode")
    with pytest.raises(ValueError, match="CEREBUS_MODE"):
        build_cerebus_completion_kwargs()


def test_build_kwargs_gateway_urls_default_when_unset(monkeypatch):
    """Both gateway URLs fall back to this org's shared nonprod endpoints
    when not overridden — no CEREBUS_GATEWAY_*_URL needed for the common case."""
    monkeypatch.setenv("CEREBUS_API_KEY", "k")
    monkeypatch.delenv("CEREBUS_GATEWAY_DIRECT_URL", raising=False)
    monkeypatch.setenv("CEREBUS_MODE", "direct")
    result = build_cerebus_completion_kwargs()
    assert result["api_base"] == "https://gw.nonprod.cerebus.tio.elsevier.systems/v1"

    monkeypatch.delenv("CEREBUS_GATEWAY_AZURE_URL", raising=False)
    monkeypatch.setenv("CEREBUS_MODE", "azure")
    monkeypatch.setenv("CEREBUS_CONFIG_ID", "pc-abc123")
    result = build_cerebus_completion_kwargs()
    assert result["api_base"] == "https://gw.az.nonprod.cerebus.tio.elsevier.systems/v1"


def test_build_kwargs_azure_mode_requires_config_id(monkeypatch):
    monkeypatch.setenv("CEREBUS_API_KEY", "k")
    monkeypatch.setenv("CEREBUS_MODE", "azure")
    monkeypatch.delenv("CEREBUS_CONFIG_ID", raising=False)
    with pytest.raises(ValueError, match="CEREBUS_CONFIG_ID"):
        build_cerebus_completion_kwargs()


def test_build_kwargs_azure_mode_success(monkeypatch):
    monkeypatch.setenv("CEREBUS_API_KEY", "k")
    monkeypatch.setenv("CEREBUS_MODE", "azure")
    monkeypatch.setenv("CEREBUS_GATEWAY_AZURE_URL", "https://gw.example/v1")
    monkeypatch.setenv("CEREBUS_CONFIG_ID", "pc-abc123")
    result = build_cerebus_completion_kwargs()
    assert result["api_base"] == "https://gw.example/v1"
    assert result["api_key"] == "k"
    assert result["extra_headers"] == {"x-portkey-api-key": "k", "x-portkey-config": "pc-abc123"}


def test_build_kwargs_direct_mode_success(monkeypatch):
    monkeypatch.setenv("CEREBUS_API_KEY", "k")
    monkeypatch.setenv("CEREBUS_MODE", "direct")
    monkeypatch.setenv("CEREBUS_GATEWAY_DIRECT_URL", "https://gw.example/v1")
    result = build_cerebus_completion_kwargs()
    assert result["api_base"] == "https://gw.example/v1"
    assert result["extra_headers"] == {"x-portkey-api-key": "k", "x-portkey-provider": "openai"}
    assert "x-portkey-config" not in result["extra_headers"]


# ---------------------------------------------------------------------------
# cerebus_model_id
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("gpt-4o-mini", "openai/gpt-4o-mini"),
        ("@sciencedirect-global-openai/gpt-5.1-chat-latest", "openai/@sciencedirect-global-openai/gpt-5.1-chat-latest"),
        ("openai/gpt-4o-mini", "openai/gpt-4o-mini"),  # idempotent
    ],
)
def test_cerebus_model_id(raw, expected):
    assert cerebus_model_id(raw) == expected


@pytest.mark.parametrize(
    "value,expected",
    [("cerebus", True), ("Cerebus", True), (" CEREBUS ", True), ("azure", False), ("", False), (None, False)],
)
def test_cerebus_enabled_via_env(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("DEFAULT_LLM_PROVIDER", raising=False)
    else:
        monkeypatch.setenv("DEFAULT_LLM_PROVIDER", value)
    assert cerebus_enabled_via_env() is expected


# ---------------------------------------------------------------------------
# Classifier: api_key / extra_headers passthrough
# ---------------------------------------------------------------------------


def _classifier(**overrides):
    from query_classification.schema import build_classification_model
    from query_classification.categories import Category, Label

    cat = Category(name="c", description="d", labels=[Label(value="x", description="d")])
    model = build_classification_model([cat])
    kwargs = dict(model_id="openai/gpt-4o-mini", system_prompt="sp", classification_model=model)
    kwargs.update(overrides)
    return Classifier(**kwargs)


def test_classifier_completion_kwargs_omit_gateway_fields_by_default():
    clf = _classifier()
    kwargs = clf._completion_kwargs([])
    assert "api_key" not in kwargs
    assert "extra_headers" not in kwargs


def test_classifier_completion_kwargs_include_gateway_fields_when_set():
    clf = _classifier(api_key="k", extra_headers={"x-portkey-api-key": "k"})
    kwargs = clf._completion_kwargs([])
    assert kwargs["api_key"] == "k"
    assert kwargs["extra_headers"] == {"x-portkey-api-key": "k"}


# ---------------------------------------------------------------------------
# Classifier: temperature dropped and retried once on UnsupportedParamsError
# (reasoning models that reject an explicit temperature entirely)
# ---------------------------------------------------------------------------


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


def _valid_content():
    import json

    return json.dumps({"c": ["x"]})


def test_complete_drops_temperature_once_on_unsupported_params_error(monkeypatch):
    import litellm

    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        if "temperature" in kwargs:
            raise litellm.UnsupportedParamsError(
                message="doesn't support temperature=0.7 while reasoning is active",
                model=kwargs.get("model", ""),
                llm_provider="openai",
            )
        return _FakeResponse(_valid_content())

    monkeypatch.setattr(litellm, "completion", fake_completion)
    clf = _classifier(temperature=0.7)
    content = clf._complete([{"role": "user", "content": "hi"}])
    assert content == _valid_content()
    assert len(calls) == 2  # one failed attempt (with temperature), one retry (without)
    assert "temperature" in calls[0]
    assert "temperature" not in calls[1]


def test_complete_reraises_unsupported_params_error_when_no_temperature_to_drop(monkeypatch):
    """If the model rejects some OTHER param we don't control, there's nothing
    to drop and retry — this must not loop or swallow the real error."""
    import litellm

    def fake_completion(**kwargs):
        raise litellm.UnsupportedParamsError(
            message="doesn't support tool_choice", model="", llm_provider="openai"
        )

    monkeypatch.setattr(litellm, "completion", fake_completion)
    clf = _classifier()  # no temperature set at all
    with pytest.raises(litellm.UnsupportedParamsError):
        clf._complete([{"role": "user", "content": "hi"}])


def test_complete_does_not_retry_temperature_drop_twice(monkeypatch):
    """A second UnsupportedParamsError after the temperature is already
    dropped must propagate, not recurse forever."""
    import litellm

    def fake_completion(**kwargs):
        raise litellm.UnsupportedParamsError(
            message="still unsupported for another reason", model="", llm_provider="openai"
        )

    monkeypatch.setattr(litellm, "completion", fake_completion)
    clf = _classifier(temperature=0.7)
    with pytest.raises(litellm.UnsupportedParamsError):
        clf._complete([{"role": "user", "content": "hi"}])


# ---------------------------------------------------------------------------
# CLI wiring: --cerebus is a real flag on both entry points, and a
# misconfigured gateway fails cleanly (no crash) rather than silently
# ignoring --cerebus or leaking a raw traceback.
# ---------------------------------------------------------------------------


def test_cli_parser_has_cerebus_flag():
    from query_classification.cli import build_parser as build_cli_parser

    ns = build_cli_parser().parse_args(
        ["--input", "i.csv", "--column", "text", "--cerebus"]
    )
    assert ns.cerebus is True


def test_experiment_parser_has_cerebus_flag_on_all_subcommands():
    from query_classification.experiment import build_parser as build_experiment_parser

    parser = build_experiment_parser()
    for subcommand, extra in (
        ("induce", ["--category-name", "c"]),
        ("classify", ["--categories", "c.json"]),
        ("run", ["--category-name", "c"]),
    ):
        ns = parser.parse_args(
            [subcommand, "--text-column", "text", "--label-column", "label",
             "--run-dir", "d", "--cerebus"] + extra
        )
        assert ns.cerebus is True


def test_cli_main_reports_clean_error_on_invalid_cerebus_mode(tmp_path, monkeypatch, capsys):
    import sys as _sys
    from query_classification.cli import main as cli_main

    # CEREBUS_MODE is optional (defaults to "direct") — only an explicitly
    # *invalid* value is an error now. no-op load_dotenv so the real .env
    # can't clobber this test's own env (see _set_direct_mode_env's comment).
    monkeypatch.setattr("query_classification.cli.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("CEREBUS_MODE", "not-a-real-mode")
    (tmp_path / "in.csv").write_text("text\nhello\n")
    _sys.argv = [
        "classify.py", "--input", str(tmp_path / "in.csv"), "--column", "text",
        "--output", str(tmp_path / "out.csv"), "--cerebus",
    ]
    with pytest.raises(SystemExit) as exc_info:
        cli_main()
    assert exc_info.value.code == 1
    assert "CEREBUS_MODE" in capsys.readouterr().out


def test_experiment_main_reports_clean_error_on_invalid_cerebus_mode(tmp_path, monkeypatch, capsys):
    import sys as _sys
    from query_classification.experiment import main as experiment_main

    monkeypatch.setattr("query_classification.experiment.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("CEREBUS_MODE", "not-a-real-mode")
    (tmp_path / "test.csv").write_text("text,label\nhello,positive\n")
    cats_path = tmp_path / "cats.json"
    cats_path.write_text(
        '{"categories": [{"name": "c", "description": "d", '
        '"labels": [{"value": "positive", "description": "d"}]}]}'
    )
    _sys.argv = [
        "experiment.py", "classify", "--test-file", str(tmp_path / "test.csv"),
        "--text-column", "text", "--label-column", "label",
        "--categories", str(cats_path), "--run-dir", str(tmp_path / "r"), "--cerebus",
    ]
    with pytest.raises(SystemExit) as exc_info:
        experiment_main()
    assert exc_info.value.code == 1
    assert "CEREBUS_MODE" in capsys.readouterr().out
    assert not (tmp_path / "r").exists()  # failed before any run-dir work


@pytest.fixture
def recorded_classifier_inits(monkeypatch):
    """Spy on every Classifier() construction: record (model_id, api_key,
    extra_headers) without changing real behavior, and monkeypatch .classify
    so no real LLM call happens. This is what FR-1.3/FR-1.5's end-to-end
    Verify conditions need — the unit-level cerebus_model_id() tests above
    don't confirm the function is actually wired into every construction
    site in cli.py/experiment.py."""
    calls = []
    original_init = Classifier.__init__

    def _spy_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        calls.append(
            {"model_id": self.model_id, "api_key": self.api_key, "extra_headers": self.extra_headers}
        )

    def _fake_classify(self, text):
        fields = set(self.classification_model.model_fields.keys())
        if fields == {"challenges", "proposed_label", "argument"}:
            return {"challenges": False, "proposed_label": None, "argument": "no challenge"}
        if "category_description" in fields:
            # Induction role (spec 2's positional description_i fields, one
            # per label position -- variable count, no fixed field set).
            return {f: "d" for f in fields}
        cat_name = next(iter(fields))
        return {cat_name: ["x"]}

    monkeypatch.setattr(Classifier, "__init__", _spy_init)
    monkeypatch.setattr(Classifier, "classify", _fake_classify)
    return calls


def _set_direct_mode_env(monkeypatch):
    # main() calls load_dotenv(override=True), which re-reads the real .env
    # on disk and would clobber these with whatever's actually there (e.g.
    # the empty CEREBUS_MODE= placeholder this feature's own setup appended)
    # — no-op it so the monkeypatched env vars below are what main() sees.
    monkeypatch.setattr("query_classification.cli.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("query_classification.experiment.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("CEREBUS_MODE", "direct")
    monkeypatch.setenv("CEREBUS_GATEWAY_DIRECT_URL", "https://gw.example/v1")
    monkeypatch.setenv("CEREBUS_API_KEY", "test-key")


def _set_minimal_footprint_env(monkeypatch):
    """Exactly the two env vars a real minimal setup uses: DEFAULT_LLM_PROVIDER
    and CEREBUS_API_KEY. No --cerebus flag, no CEREBUS_MODE, no gateway URL —
    everything else must come from the built-in defaults."""
    monkeypatch.setattr("query_classification.cli.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("query_classification.experiment.load_dotenv", lambda *a, **k: None)
    for var in ("CEREBUS_MODE", "CEREBUS_GATEWAY_AZURE_URL", "CEREBUS_GATEWAY_DIRECT_URL", "CEREBUS_CONFIG_ID"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "cerebus")
    monkeypatch.setenv("CEREBUS_API_KEY", "test-key")


def test_cli_minimal_footprint_env_only(tmp_path, monkeypatch, recorded_classifier_inits):
    """The exact setup the user asked to replicate: DEFAULT_LLM_PROVIDER=cerebus
    + CEREBUS_API_KEY, nothing else, no --cerebus flag — must route through
    Cerebus using the built-in direct-mode gateway default."""
    import sys as _sys
    from query_classification.cli import main as cli_main

    _set_minimal_footprint_env(monkeypatch)
    (tmp_path / "in.csv").write_text("text\nhello\n")
    cats_path = tmp_path / "cats.json"
    cats_path.write_text(
        '{"categories": [{"name": "c", "description": "d", '
        '"labels": [{"value": "x", "description": "d"}]}]}'
    )
    _sys.argv = [
        "classify.py", "--input", str(tmp_path / "in.csv"), "--column", "text",
        "--output", str(tmp_path / "out.csv"), "--categories", str(cats_path),
    ]
    cli_main()

    calls = recorded_classifier_inits
    assert len(calls) == 1
    assert calls[0]["model_id"].startswith("openai/")
    assert calls[0]["extra_headers"] == {
        "x-portkey-api-key": "test-key",
        "x-portkey-provider": "openai",
    }


def test_experiment_minimal_footprint_env_only(tmp_path, monkeypatch, recorded_classifier_inits):
    import sys as _sys
    from query_classification.experiment import main as experiment_main

    _set_minimal_footprint_env(monkeypatch)
    test = tmp_path / "test.csv"
    test.write_text("text,label\nhello,positive\n")
    cats_path = tmp_path / "cats.json"
    cats_path.write_text(
        '{"categories": [{"name": "c", "description": "d", '
        '"labels": [{"value": "positive", "description": "d"}]}]}'
    )
    run_dir = tmp_path / "r"
    _sys.argv = [
        "experiment.py", "classify", "--test-file", str(test), "--text-column", "text",
        "--label-column", "label", "--categories", str(cats_path), "--run-dir", str(run_dir),
    ]
    experiment_main()

    calls = recorded_classifier_inits
    assert len(calls) == 1
    assert calls[0]["model_id"].startswith("openai/")
    assert calls[0]["extra_headers"]["x-portkey-provider"] == "openai"


def test_cli_critics_run_prefixes_and_shares_headers_across_every_role(
    tmp_path, monkeypatch, recorded_classifier_inits
):
    """FR-1.3 + FR-1.5, cli.py side: every role's model id gets the openai/
    prefix, and every role shares identical extra_headers."""
    import sys as _sys
    from query_classification.cli import main as cli_main

    _set_direct_mode_env(monkeypatch)
    (tmp_path / "in.csv").write_text("text\nhello\n")
    cats_path = tmp_path / "cats.json"
    cats_path.write_text(
        '{"categories": [{"name": "c", "description": "d", '
        '"labels": [{"value": "x", "description": "d"}]}]}'
    )
    _sys.argv = [
        "classify.py", "--input", str(tmp_path / "in.csv"), "--column", "text",
        "--output", str(tmp_path / "out.csv"), "--categories", str(cats_path),
        "--cerebus", "--critics",
        "--model", "base-model", "--critic-model", "critic-model", "--reconciler-model", "reconciler-model",
    ]
    cli_main()

    calls = recorded_classifier_inits
    assert len(calls) == 3  # classification, critic, reconciler
    model_ids = {c["model_id"] for c in calls}
    assert model_ids == {"openai/base-model", "openai/critic-model", "openai/reconciler-model"}
    headers = [c["extra_headers"] for c in calls]
    assert all(h == headers[0] for h in headers)
    assert headers[0] == {"x-portkey-api-key": "test-key", "x-portkey-provider": "openai"}
    assert all(c["api_key"] == "test-key" for c in calls)


def test_experiment_run_prefixes_and_shares_headers_across_every_role(
    tmp_path, monkeypatch, recorded_classifier_inits
):
    """FR-1.3 + FR-1.5, experiment.py side: the induction role and the
    classification role both get prefixed model ids and identical headers."""
    import sys as _sys
    from query_classification.experiment import main as experiment_main

    _set_direct_mode_env(monkeypatch)
    train = tmp_path / "train.csv"
    train.write_text("text,label\n" + "\n".join(f'"row{i}","x"' for i in range(3)) + "\n")
    test = tmp_path / "test.csv"
    test.write_text('text,label\n"row","x"\n')
    run_dir = tmp_path / "r"
    _sys.argv = [
        "experiment.py", "run", "--train-file", str(train), "--test-file", str(test),
        "--text-column", "text", "--label-column", "label", "--category-name", "c",
        "--run-dir", str(run_dir), "--cerebus",
        "--model", "base-model", "--induction-model", "induction-model",
    ]
    try:
        experiment_main()
    except SystemExit as e:
        assert e.code in (0, None)

    calls = recorded_classifier_inits
    assert len(calls) == 2  # induction, classification
    model_ids = {c["model_id"] for c in calls}
    assert model_ids == {"openai/induction-model", "openai/base-model"}
    headers = [c["extra_headers"] for c in calls]
    assert all(h == headers[0] for h in headers)
    assert headers[0] == {"x-portkey-api-key": "test-key", "x-portkey-provider": "openai"}


# ---------------------------------------------------------------------------
# --cerebus defaults to False; an --api-base override must be HTTPS
# ---------------------------------------------------------------------------


def test_cli_parser_cerebus_defaults_false():
    from query_classification.cli import build_parser as build_cli_parser

    ns = build_cli_parser().parse_args(["--input", "i.csv", "--column", "text"])
    assert ns.cerebus is False


def test_experiment_parser_cerebus_defaults_false():
    from query_classification.experiment import build_parser as build_experiment_parser

    ns = build_experiment_parser().parse_args(
        ["classify", "--text-column", "text", "--categories", "c.json", "--run-dir", "d"]
    )
    assert ns.cerebus is False


def test_reject_insecure_cerebus_endpoint():
    from query_classification.classifier import reject_insecure_cerebus_endpoint

    reject_insecure_cerebus_endpoint(None)  # no override at all: fine
    reject_insecure_cerebus_endpoint("https://gw.example/v1")  # HTTPS: fine
    with pytest.raises(ValueError, match="https://"):
        reject_insecure_cerebus_endpoint("http://gw.example/v1")


def test_cli_main_rejects_insecure_api_base_override_under_cerebus(tmp_path, monkeypatch, capsys):
    import sys as _sys
    from query_classification.cli import main as cli_main

    _set_direct_mode_env(monkeypatch)
    (tmp_path / "in.csv").write_text("text\nhello\n")
    _sys.argv = [
        "classify.py", "--input", str(tmp_path / "in.csv"), "--column", "text",
        "--output", str(tmp_path / "out.csv"), "--cerebus",
        "--api-base", "http://insecure.example/v1",
    ]
    with pytest.raises(SystemExit) as exc_info:
        cli_main()
    assert exc_info.value.code == 1
    assert "https://" in capsys.readouterr().out
    assert not (tmp_path / "out.csv").exists()


def test_experiment_main_rejects_insecure_api_base_override_under_cerebus(tmp_path, monkeypatch, capsys):
    import sys as _sys
    from query_classification.experiment import main as experiment_main

    _set_direct_mode_env(monkeypatch)
    (tmp_path / "test.csv").write_text("text,label\nhello,positive\n")
    cats_path = tmp_path / "cats.json"
    cats_path.write_text(
        '{"categories": [{"name": "c", "description": "d", '
        '"labels": [{"value": "positive", "description": "d"}]}]}'
    )
    run_dir = tmp_path / "r"
    _sys.argv = [
        "experiment.py", "classify", "--test-file", str(tmp_path / "test.csv"),
        "--text-column", "text", "--label-column", "label",
        "--categories", str(cats_path), "--run-dir", str(run_dir), "--cerebus",
        "--api-base", "http://insecure.example/v1",
    ]
    with pytest.raises(SystemExit) as exc_info:
        experiment_main()
    assert exc_info.value.code == 1
    assert "https://" in capsys.readouterr().out
    assert not run_dir.exists()


# ---------------------------------------------------------------------------
# .env is never modified by any resolution path (FR-2.5)
# ---------------------------------------------------------------------------


def test_env_file_never_modified_by_aws_fallback_resolution(tmp_path, monkeypatch, fake_boto3):
    env_path = tmp_path / ".env"
    env_path.write_text("SOME_OTHER_VAR=1\n")
    before = env_path.read_bytes()
    monkeypatch.delenv("CEREBUS_API_KEY", raising=False)
    classifier_module._resolve_cerebus_api_key()
    assert env_path.read_bytes() == before
