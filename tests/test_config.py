import pytest

from regression_intelligence.utils.config import _deep_merge, load_config


def test_dev_overrides_base():
    cfg = load_config("dev")
    assert cfg.training.cv_folds == 3
    assert cfg.project.random_seed == 42


def test_production_uses_base_cv():
    assert load_config("production").training.cv_folds == 5


def test_unknown_env_raises():
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent")


def test_deep_merge_keeps_untouched_keys():
    merged = _deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 9}})
    assert merged == {"a": {"x": 1, "y": 9}}
