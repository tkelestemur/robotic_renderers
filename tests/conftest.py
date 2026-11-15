"""Keep default-host assertions independent of the developer's renderer setup."""

import pytest


def pytest_configure(config):
    patch = pytest.MonkeyPatch()
    config.add_cleanup(patch.undo)
    patch.setenv("ROBOTIC_RENDERERS_HOST", "render-host")
