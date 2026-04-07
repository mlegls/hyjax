"""Pytest collector for Hy test files.

Importing ``hy`` installs an import hook that lets Python import ``.hy``
modules as if they were ``.py``. We then register a ``pytest_collect_file``
hook so pytest will treat ``test_*.hy`` files the same way it treats
``test_*.py``: collect module-level ``test_*`` functions as test items.

Hy mangles kebab-case names to snake_case, so ``(defn test-foo [] ...)``
defined in a ``.hy`` file is exposed to Python as ``test_foo`` and will
be picked up by pytest's default naming convention.
"""

import hy  # noqa: F401 -- registers the .hy import hook
from _pytest.python import Module


def pytest_collect_file(parent, file_path):
    if file_path.suffix == ".hy" and file_path.name.startswith("test_"):
        return Module.from_parent(parent, path=file_path)
