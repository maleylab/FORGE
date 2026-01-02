# tests/plans/test_planentry_schema.py

import pytest

from labtools.plans.schema import validate_planentry_dict


def _valid_planentry_dict():
    return {
        "id": "test_001",
        "schema": {"name": "test_schema", "version": 1},
        "intent": {"task": "noop", "system": "dummy"},
        "parameters": {"x": 1, "y": 2},
        "metadata": {"tags": ["a", "b"], "notes": "ok"},
    }


def test_valid_planentry_passes():
    validate_planentry_dict(_valid_planentry_dict())


def test_missing_top_level_field_fails():
    d = _valid_planentry_dict()
    del d["intent"]
    with pytest.raises(ValueError, match="Missing PlanEntry field"):
        validate_planentry_dict(d)


def test_missing_schema_version_fails():
    d = _valid_planentry_dict()
    del d["schema"]["version"]
    with pytest.raises(ValueError, match="schema must contain"):
        validate_planentry_dict(d)


def test_missing_intent_system_fails():
    d = _valid_planentry_dict()
    del d["intent"]["system"]
    with pytest.raises(ValueError, match="intent must contain"):
        validate_planentry_dict(d)


def test_parameters_must_be_dict():
    d = _valid_planentry_dict()
    d["parameters"] = ["not", "a", "dict"]
    with pytest.raises(ValueError, match="parameters must be a dict"):
        validate_planentry_dict(d)


def test_tags_must_be_list_of_strings():
    d = _valid_planentry_dict()
    d["metadata"]["tags"] = [1, 2, 3]
    with pytest.raises(ValueError, match="metadata.tags"):
        validate_planentry_dict(d)

