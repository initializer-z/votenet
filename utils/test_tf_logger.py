# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' Tests for utils/tf_logger.py — focused on the import os deduplication change. '''

import os
import sys
import types
import importlib
import unittest
from unittest.mock import MagicMock

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(BASE_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)


def _mock_heavy_deps():
    """Insert lightweight mocks for tensorflow and scipy into sys.modules so that
    importing tf_logger does not require those packages to be installed."""
    # Mock tensorflow
    if 'tensorflow' not in sys.modules:
        tf_mock = MagicMock()
        tf_mock.__name__ = 'tensorflow'
        tf_mock.__spec__ = importlib.util.spec_from_loader('tensorflow', loader=None)
        sys.modules['tensorflow'] = tf_mock

    # Mock scipy and scipy.misc (used for toimage)
    if 'scipy' not in sys.modules:
        scipy_mock = MagicMock()
        scipy_mock.__name__ = 'scipy'
        sys.modules['scipy'] = scipy_mock
    if 'scipy.misc' not in sys.modules:
        scipy_misc_mock = MagicMock()
        scipy_misc_mock.__name__ = 'scipy.misc'
        sys.modules['scipy.misc'] = scipy_misc_mock


def _import_tf_logger_fresh():
    """Remove tf_logger from sys.modules (if present) and re-import it, returning the module."""
    mod_name = 'utils.tf_logger'
    alt_name = 'tf_logger'
    for name in (mod_name, alt_name):
        sys.modules.pop(name, None)

    _mock_heavy_deps()

    # Import via direct path to support both `python utils/test_tf_logger.py` and pytest
    spec = importlib.util.spec_from_file_location(
        'tf_logger', os.path.join(BASE_DIR, 'tf_logger.py')
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestImportOsPresence(unittest.TestCase):
    """The PR adds two extra `import os` lines (for a total of three).
    These tests verify the semantics of that change."""

    def setUp(self):
        self.tf_logger = _import_tf_logger_fresh()

    def test_os_attribute_exists_in_module(self):
        """After importing tf_logger, `os` must be a name in its namespace."""
        self.assertIn('os', vars(self.tf_logger),
                      "tf_logger module should export 'os' in its namespace")

    def test_os_is_the_real_os_module(self):
        """The `os` name in tf_logger must refer to the standard library os module,
        not None or a mock."""
        module_os = vars(self.tf_logger)['os']
        self.assertIsInstance(module_os, types.ModuleType,
                              "'os' in tf_logger should be a module instance")
        self.assertEqual(module_os.__name__, 'os',
                         "'os' in tf_logger should be the os standard library module")

    def test_os_identity_matches_sys_modules(self):
        """The `os` object in tf_logger's namespace must be the same object stored
        in sys.modules['os'] — duplicate imports must not create a fresh copy."""
        module_os = vars(self.tf_logger)['os']
        self.assertIs(module_os, sys.modules['os'],
                      "Duplicate `import os` must not create a separate os object; "
                      "it should be identical to sys.modules['os']")

    def test_os_path_attr_accessible(self):
        """os.path must be accessible through tf_logger's os reference, confirming
        the import resolved to a fully-initialised os module."""
        module_os = vars(self.tf_logger)['os']
        self.assertTrue(hasattr(module_os, 'path'),
                        "os module in tf_logger should have the 'path' sub-module")

    def test_os_getcwd_callable(self):
        """os.getcwd must be callable through tf_logger's os reference — a basic
        smoke-test that the resolved module is functional."""
        module_os = vars(self.tf_logger)['os']
        self.assertTrue(callable(getattr(module_os, 'getcwd', None)),
                        "os.getcwd should be callable via tf_logger's os reference")


class TestDuplicateImportIdempotency(unittest.TestCase):
    """Regression tests: multiple imports of tf_logger must not alter the os binding."""

    def test_reimport_preserves_os_identity(self):
        """Importing tf_logger a second time should yield the same os object as
        the first import — sys.modules caching must not be broken."""
        first = _import_tf_logger_fresh()
        os_first = vars(first)['os']

        second = _import_tf_logger_fresh()
        os_second = vars(second)['os']

        self.assertIs(os_first, os_second,
                      "Re-importing tf_logger should not produce a different os object")

    def test_os_in_sys_modules_unchanged_after_import(self):
        """After importing tf_logger (with its three `import os` lines), the os entry
        in sys.modules should remain the standard library module."""
        _import_tf_logger_fresh()
        self.assertIn('os', sys.modules,
                      "sys.modules['os'] must exist after importing tf_logger")
        self.assertEqual(sys.modules['os'].__name__, 'os',
                         "sys.modules['os'] must still be the real os module after tf_logger import")

    def test_no_extra_os_keys_in_sys_modules(self):
        """Three `import os` statements must not leave additional aliased keys for os
        in sys.modules (e.g. no 'os.os' or duplicate entries)."""
        _import_tf_logger_fresh()
        os_keys = [k for k in sys.modules if k == 'os' or k.startswith('os.')]
        # Only 'os' and optional 'os.path' are expected standard entries
        unexpected = [k for k in os_keys if k not in ('os', 'os.path')]
        self.assertEqual(unexpected, [],
                         f"Unexpected sys.modules keys related to 'os': {unexpected}")


class TestModuleImportDoesNotRaise(unittest.TestCase):
    """Boundary / negative tests: the import must succeed without exceptions."""

    def test_import_does_not_raise(self):
        """Importing tf_logger (including its triplicate import os) must not raise."""
        try:
            _import_tf_logger_fresh()
        except Exception as exc:
            self.fail(f"Importing tf_logger raised an unexpected exception: {exc}")

    def test_import_returns_module_type(self):
        """The object returned by importing tf_logger must be a Python module."""
        tf_logger = _import_tf_logger_fresh()
        self.assertIsInstance(tf_logger, types.ModuleType,
                              "tf_logger import should return a module object")


if __name__ == '__main__':
    unittest.main()
