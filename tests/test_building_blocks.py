"""Offline tests for the schema/prompt building blocks (no network calls)."""

import json

import pytest

from query_classification import (
    build_classification_model,
    build_system_prompt,
    load_categories,
)
from query_classification.prompts import build_critic_prompt, build_reconciler_prompt
from query_classification.resources import (
    DEFAULT_CATEGORIES_FILE,
    DEFAULT_CRITIC_PROMPT_FILE,
    DEFAULT_RECONCILER_PROMPT_FILE,
    DEFAULT_SYSTEM_PROMPT_FILE,
    EXAMPLE_TASK_DESCRIPTION_FILE,
)


@pytest.fixture
def categories(tmp_path):
    data = {
        "categories": [
            {
                "name": "sentiment",
                "description": "Overall sentiment.",
                "labels": [
                    {"value": "positive", "description": "Happy."},
                    {"value": "negative", "description": "Sad."},
                ],
            }
        ]
    }
    path = tmp_path / "cats.json"
    path.write_text(json.dumps(data))
    return load_categories(path)


def test_load_categories(categories):
    assert len(categories) == 1
    assert categories[0].name == "sentiment"
    assert [label.value for label in categories[0].labels] == ["positive", "negative"]


def test_build_classification_model_validates_label_lists(categories):
    model = build_classification_model(categories)
    assert "sentiment" in model.model_fields

    parsed = model.model_validate_json('{"sentiment": ["positive"]}')
    assert parsed.model_dump() == {"sentiment": ["positive"]}

    # Enforces the 1-3 label bound.
    with pytest.raises(Exception):
        model.model_validate_json('{"sentiment": ["a", "b", "c", "d"]}')


def test_build_system_prompt_fills_slots(categories):
    model = build_classification_model(categories)
    prompt = build_system_prompt(
        model,
        task_description="Classify movie reviews.",
        extra_prompt="Be strict.",
    )
    assert "Classify movie reviews." in prompt
    assert "sentiment" in prompt
    assert "Additional instructions:" in prompt
    assert "Be strict." in prompt
    # No unfilled template placeholders remain.
    assert "{task_description}" not in prompt
    assert "{schema_description}" not in prompt


def test_bundled_resources_exist():
    assert DEFAULT_SYSTEM_PROMPT_FILE.exists()
    assert DEFAULT_CATEGORIES_FILE.exists()
    assert EXAMPLE_TASK_DESCRIPTION_FILE.exists()
    assert DEFAULT_CRITIC_PROMPT_FILE.exists()
    assert DEFAULT_RECONCILER_PROMPT_FILE.exists()
    # The bundled categories file loads and builds a model cleanly.
    build_classification_model(load_categories(DEFAULT_CATEGORIES_FILE))


def test_build_system_prompt_allow_new_labels_toggle(categories):
    default_model = build_classification_model(categories)
    default_prompt = build_system_prompt(default_model)

    # Both halves of the constraint (schema field description + system prompt
    # template) must be built with the same allow_new_labels value.
    constrained_model = build_classification_model(categories, allow_new_labels=False)
    no_invention_prompt = build_system_prompt(constrained_model, allow_new_labels=False)

    assert "none - " in default_prompt
    assert "none - " not in no_invention_prompt
    assert "{invention_policy}" not in default_prompt
    assert "{invention_policy}" not in no_invention_prompt


def test_build_critic_and_reconciler_prompts():
    critic_prompt = build_critic_prompt("sentiment", "Overall sentiment.", '"positive": Happy.')
    assert "sentiment" in critic_prompt
    assert "Overall sentiment." in critic_prompt
    assert "positive" in critic_prompt

    reconciler_prompt = build_reconciler_prompt(
        "sentiment", "Overall sentiment.", '"positive": Happy.'
    )
    assert "sentiment" in reconciler_prompt
    assert "Overall sentiment." in reconciler_prompt
    assert "positive" in reconciler_prompt
