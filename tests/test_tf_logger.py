# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for utils/tf_logger.py

The PR added duplicate `import os` statements to the module (lines 9-11).
These tests verify that:
  - Python's module caching handles duplicate `import os` correctly
  - The `os` module is accessible and functional after duplicate imports
  - The source file reflects the added duplicate imports (regression guard)
  - The module can be loaded when its heavy dependencies are mocked
"""

import ast
import importlib.util
import os
import sys
import unittest
from unittest import mock

_UTILS_DIR = os.path.join(os.path.dirname(__file__), "..", "utils")
_TF_LOGGER_PATH = os.path.join(_UTILS_DIR, "tf_logger.py")


def _load_tf_logger_with_mocks(tf_mock=None, scipy_mock=None):
    """Load utils/tf_logger.py via importlib with heavy deps mocked in sys.modules."""
    if tf_mock is None:
        tf_mock = mock.MagicMock()
        tf_mock.summary.FileWriter = mock.MagicMock()
        tf_mock.Summary = mock.MagicMock()
        tf_mock.HistogramProto = mock.MagicMock()
    if scipy_mock is None:
        scipy_mock = mock.MagicMock()

    patches = {
        "tensorflow": tf_mock,
        "scipy": scipy_mock,
        "scipy.misc": scipy_mock.misc,
    }

    spec = importlib.util.spec_from_file_location("tf_logger", _TF_LOGGER_PATH)
    module = importlib.util.module_from_spec(spec)

    with mock.patch.dict(sys.modules, patches):
        spec.loader.exec_module(module)

    return module


class TestDuplicateOsImportBehavior(unittest.TestCase):
    """Tests verifying Python's behavior with duplicate `import os` statements.

    The PR introduced two additional `import os` lines into utils/tf_logger.py.
    In Python, re-importing an already-imported module is a no-op: the runtime
    looks up the cached entry in sys.modules and binds the same object to the
    name again. These tests document and assert that contract.
    """

    def test_duplicate_import_os_returns_same_object(self):
        """Duplicate `import os` statements must resolve to the same module object."""
        import os as os_first
        import os as os_second
        import os as os_third  # mirrors the three `import os` lines in tf_logger.py
        self.assertIs(os_first, os_second)
        self.assertIs(os_second, os_third)

    def test_duplicate_import_os_does_not_duplicate_sys_modules_entry(self):
        """sys.modules['os'] must remain a single entry regardless of duplicate imports."""
        import os as _os  # noqa: F401
        entry_before = sys.modules.get("os")
        import os  # duplicate – same as what tf_logger.py does twice more  # noqa: F811
        import os  # duplicate  # noqa: F811
        entry_after = sys.modules.get("os")
        self.assertIs(entry_before, entry_after)

    def test_os_module_is_standard_library_os(self):
        """The `os` module bound by any of the duplicate imports is the real stdlib os."""
        import os as imported_os
        self.assertEqual(imported_os.__name__, "os")
        # Spot-check well-known attributes to confirm it is the genuine module.
        self.assertTrue(hasattr(imported_os, "path"))
        self.assertTrue(hasattr(imported_os, "environ"))
        self.assertTrue(hasattr(imported_os, "getcwd"))

    def test_os_identity_preserved_after_repeated_imports(self):
        """Module identity (id()) of `os` must be stable across repeated imports."""
        import os as ref
        ref_id = id(ref)
        import os  # first extra import (mirrors tf_logger.py line 10)  # noqa: F811
        import os  # second extra import (mirrors tf_logger.py line 11)  # noqa: F811
        import os as after  # noqa: F811
        self.assertEqual(id(after), ref_id)

    def test_os_path_join_functional_after_duplicate_imports(self):
        """os.path.join must work correctly after duplicate imports, showing os is intact."""
        import os as _os
        import os  # duplicate  # noqa: F811
        import os  # duplicate  # noqa: F811
        result = _os.path.join("some", "directory", "file.txt")
        self.assertEqual(result, os.path.join("some", "directory", "file.txt"))

    def test_os_environ_accessible_after_duplicate_imports(self):
        """os.environ must be accessible and usable after the duplicate imports."""
        import os as _os
        import os  # duplicate  # noqa: F811
        import os  # duplicate  # noqa: F811
        self.assertIsInstance(_os.environ, os._Environ)

    def test_os_getcwd_returns_string_after_duplicate_imports(self):
        """os.getcwd() must return a non-empty string, verifying os functionality."""
        import os as _os
        import os  # duplicate  # noqa: F811
        import os  # duplicate  # noqa: F811
        cwd = _os.getcwd()
        self.assertIsInstance(cwd, str)
        self.assertTrue(len(cwd) > 0)


class TestTfLoggerSourceContainsDuplicateImports(unittest.TestCase):
    """Regression tests that guard the PR's source-level changes.

    These parse the actual source file with the `ast` module to verify that
    the three `import os` statements (the original plus the two added by this PR)
    are present. If a future cleanup removes them, these tests will catch it.
    """

    def _get_import_os_nodes(self):
        with open(_TF_LOGGER_PATH, "r") as fh:
            source = fh.read()
        tree = ast.parse(source)
        nodes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "os":
                        nodes.append(node)
        return nodes

    def test_source_file_readable(self):
        """utils/tf_logger.py must exist and be readable."""
        self.assertTrue(os.path.isfile(_TF_LOGGER_PATH))

    def test_source_contains_at_least_three_import_os_statements(self):
        """The PR added two extra `import os` lines; the file must have >= 3 total."""
        nodes = self._get_import_os_nodes()
        self.assertGreaterEqual(
            len(nodes),
            3,
            msg=(
                "Expected at least 3 'import os' statements (original + 2 added by PR), "
                f"found {len(nodes)}."
            ),
        )

    def test_source_contains_exactly_three_import_os_statements(self):
        """Exact regression guard: the file must have exactly 3 `import os` lines."""
        nodes = self._get_import_os_nodes()
        self.assertEqual(
            len(nodes),
            3,
            msg=(
                f"Expected exactly 3 'import os' statements, found {len(nodes)}. "
                "This test guards the duplicate imports introduced by the PR."
            ),
        )

    def test_duplicate_import_os_lines_are_consecutive(self):
        """The three `import os` statements must appear on consecutive lines."""
        nodes = self._get_import_os_nodes()
        self.assertEqual(len(nodes), 3, msg="Need exactly 3 import os nodes to check consecutiveness.")
        line_numbers = sorted(node.lineno for node in nodes)
        # They should be consecutive: e.g. [9, 10, 11].
        self.assertEqual(
            line_numbers,
            list(range(line_numbers[0], line_numbers[0] + 3)),
            msg=(
                f"Expected three consecutive 'import os' lines, got lines: {line_numbers}"
            ),
        )

    def test_source_parses_without_syntax_errors(self):
        """utils/tf_logger.py must parse cleanly as valid Python (no syntax errors)."""
        with open(_TF_LOGGER_PATH, "r") as fh:
            source = fh.read()
        try:
            ast.parse(source)
        except SyntaxError as exc:
            self.fail(f"utils/tf_logger.py contains a syntax error: {exc}")


class TestTfLoggerModuleImportWithMockedDependencies(unittest.TestCase):
    """Tests that utils/tf_logger.py can be loaded when heavy deps are mocked.

    This exercises the module-level code path (including the triple `import os`)
    by actually executing the module under a controlled environment via importlib.
    """

    def test_module_loads_without_error_when_tf_is_mocked(self):
        """utils/tf_logger.py must load cleanly when tensorflow is replaced by a mock."""
        # Should not raise any exception.
        _load_tf_logger_with_mocks()

    def test_os_is_accessible_in_module_namespace_after_load(self):
        """After loading utils/tf_logger.py, `os` must be in its namespace."""
        module = _load_tf_logger_with_mocks()
        self.assertIn("os", vars(module))

    def test_os_in_tf_logger_namespace_is_stdlib_os(self):
        """The `os` bound in the tf_logger module namespace must be the stdlib os module."""
        module = _load_tf_logger_with_mocks()
        self.assertIs(module.os, os)
        self.assertEqual(module.os.__name__, "os")
        self.assertTrue(hasattr(module.os, "path"))
        self.assertTrue(hasattr(module.os, "environ"))

    def test_logger_class_present_after_load_with_mocked_deps(self):
        """The Logger class must exist in the module after a successful load."""
        module = _load_tf_logger_with_mocks()
        self.assertTrue(hasattr(module, "Logger"))
        self.assertIsInstance(module.Logger, type)

    def test_loading_module_twice_yields_same_os_identity(self):
        """Loading the module a second time must not produce a different `os` object."""
        module_a = _load_tf_logger_with_mocks()
        module_b = _load_tf_logger_with_mocks()
        # Both should reference the same cached stdlib os module.
        self.assertIs(module_a.os, module_b.os)

    def test_os_path_operations_work_via_module_namespace(self):
        """os.path.join accessed through the loaded module's namespace must work correctly."""
        module = _load_tf_logger_with_mocks()
        result = module.os.path.join("a", "b", "c")
        self.assertEqual(result, os.path.join("a", "b", "c"))


if __name__ == "__main__":
    unittest.main()
