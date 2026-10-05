import pytest

from tests.synthetic import make_rossmann_like


@pytest.fixture(scope="session")
def clean_df():
    """Valid data. Tests must .copy() before modifying."""
    return make_rossmann_like(inject_issues=False)


@pytest.fixture(scope="session")
def dirty_df():
    """Same data with realistic problems injected."""
    return make_rossmann_like(inject_issues=True)
