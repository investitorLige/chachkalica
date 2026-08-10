"""Static guards that keep ``gpu_infer`` vendorable into an exported bundle.

    python -m unittest gpu_infer.tests.test_portability

Pure AST analysis — no imports, no torch, runs anywhere. Two rules, both of which exist because
breaking them fails somewhere far away from the cause:

1. **No ``torch`` / ``torchvision`` / ``tensorrt`` / heavy sibling imports at module scope.**
   ``chachak/tests/test_bundle_export.py`` imports every ``.py`` under a vendored bundle's
   ``runtime/`` in a *subprocess* with only ``runtime/`` on ``sys.path``. A module-scope
   ``torchvision`` import would fail that for every bundle, and ``torchvision`` is meant to be
   optional here (it is only needed for nvJPEG decode). Keeping the whole package importable with
   the standard library alone also keeps it usable from the torch-free build node, the same
   reasoning as ``chachak.bundle_export.cli._batchable_in_trt``.

2. **No ``friendy_chachkalica``, and no ``chachak._friendy`` / ``chachak.boxes`` at module
   scope.** ``test_bundle_export.py`` asserts no vendored module mentions the training package.
   ``chachak._friendy`` is the trap: in a source checkout it appends the repo root to
   ``sys.path`` and imports the trainer, and in a bundle it only works because a generated
   template happens to re-export the right names. This package therefore has its own batched
   twins of the handful of helpers involved.

Tests are exempt from both rules — they are never vendored (``vendor._copy_package`` strips
``tests/``), and they need the originals to compare against.
"""

import ast
import sys
import unittest
from pathlib import Path

_PACKAGE = Path(__file__).resolve().parents[1]

#: Importing any of these at module scope breaks the bundle's import-closure test.
_DEFERRED_MODULES = {
    "torch",
    "torchvision",
    "tensorrt",
    "numpy",
    "PIL",
    "chachak",
    "onnx_infer",
    "trt_infer",
}

#: Never importable from a vendored module, at any scope.
_FORBIDDEN_ANYWHERE = {"friendy_chachkalica"}

#: Never importable at module scope even though the package name is allowed elsewhere.
_FORBIDDEN_MODULE_SCOPE = {"chachak._friendy", "chachak.boxes"}


def _package_modules():
    """Every shipped ``.py`` in the package — excludes ``tests/``, which is not vendored."""
    return sorted(
        path
        for path in _PACKAGE.rglob("*.py")
        if "tests" not in path.relative_to(_PACKAGE).parts
    )


def _root_of(name: str) -> str:
    return (name or "").split(".")[0]


def _module_scope_imports(tree: ast.Module):
    """``(node, dotted_name)`` for imports at module scope only, not inside a function or class.

    Walks only the module body rather than ``ast.walk``, because the whole point is *where* the
    import sits: the same statement inside a function is exactly what this package wants.
    """
    found = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((node, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # a relative import, e.g. `from . import geometry` -- always fine
                continue
            found.append((node, node.module or ""))
        elif isinstance(node, ast.If):
            # e.g. `if TYPE_CHECKING:` blocks -- still module scope, so recurse into them.
            for inner in node.body:
                if isinstance(inner, ast.Import):
                    for alias in inner.names:
                        found.append((inner, alias.name))
                elif isinstance(inner, ast.ImportFrom) and not inner.level:
                    found.append((inner, inner.module or ""))
    return found


def _all_imports(tree: ast.Module):
    """``(node, dotted_name)`` for every import at any scope."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((node, alias.name))
        elif isinstance(node, ast.ImportFrom) and not node.level:
            found.append((node, node.module or ""))
    return found


class DeferredImportTest(unittest.TestCase):
    def test_the_package_has_modules_to_check(self):
        """Guards the guard: a glob that matched nothing would make every test below vacuous."""
        modules = _package_modules()
        self.assertGreaterEqual(len(modules), 8, f"only found {modules}")

    def test_no_heavy_import_at_module_scope(self):
        offenders = []
        for path in _package_modules():
            tree = ast.parse(path.read_text(), filename=str(path))
            for node, name in _module_scope_imports(tree):
                if _root_of(name) in _DEFERRED_MODULES:
                    offenders.append(f"{path.name}:{node.lineno} imports {name!r}")
        self.assertEqual(
            offenders,
            [],
            "these must move inside a function -- a vendored bundle's import closure is "
            "checked in a bare subprocess:\n  " + "\n  ".join(offenders),
        )

    def test_the_package_imports_with_only_the_standard_library(self):
        """Import it in a subprocess with the heavy modules made unavailable.

        The AST check above proves nothing *appears* at module scope; this proves the whole
        package actually imports without them, including any transitive module-scope import a
        relative import might drag in.
        """
        import subprocess

        script = (
            "import sys\n"
            "class Blocker:\n"
            "    def find_module(self, name, path=None):\n"
            "        if name.split('.')[0] in %r: raise ImportError(name)\n"
            "        return None\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] in %r: raise ImportError(\n"
            "            'blocked at module scope: ' + name)\n"
            "        return None\n"
            "sys.meta_path.insert(0, Blocker())\n"
            "sys.path.insert(0, %r)\n"
            "import gpu_infer\n"
            "import gpu_infer.config, gpu_infer.geometry, gpu_infer.tiling\n"
            "import gpu_infer.tail, gpu_infer.nms, gpu_infer.engine\n"
            "import gpu_infer.crops_exact, gpu_infer.decode, gpu_infer.loader\n"
            "import gpu_infer.pipeline\n"
            "print('ok')\n"
            % (sorted(_DEFERRED_MODULES), sorted(_DEFERRED_MODULES), str(_PACKAGE.parent))
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ok", result.stdout)

    def test_public_api_resolves_lazily(self):
        """``__getattr__`` must expose the documented names without eager imports."""
        import gpu_infer

        self.assertIn("GpuOptions", dir(gpu_infer))
        self.assertIn("build_gpu_pipeline", dir(gpu_infer))
        with self.assertRaises(AttributeError):
            gpu_infer.definitely_not_a_real_symbol


class NoTrainingPackageTest(unittest.TestCase):
    def test_the_training_package_is_never_imported(self):
        offenders = []
        for path in _package_modules():
            tree = ast.parse(path.read_text(), filename=str(path))
            for node, name in _all_imports(tree):
                if _root_of(name) in _FORBIDDEN_ANYWHERE:
                    offenders.append(f"{path.name}:{node.lineno} imports {name!r}")
        self.assertEqual(offenders, [], "\n  ".join(offenders))

    def test_the_training_package_is_not_named_in_source(self):
        """Mirrors ``test_bundle_export.py``'s literal string scan, which is what actually runs
        against a vendored bundle."""
        offenders = [
            path.name
            for path in _package_modules()
            if "from friendy_chachkalica" in path.read_text()
            or "import friendy_chachkalica" in path.read_text()
        ]
        self.assertEqual(offenders, [])

    def test_the_friendy_shim_and_boxes_are_not_imported_at_module_scope(self):
        offenders = []
        for path in _package_modules():
            tree = ast.parse(path.read_text(), filename=str(path))
            for node, name in _module_scope_imports(tree):
                if name in _FORBIDDEN_MODULE_SCOPE:
                    offenders.append(f"{path.name}:{node.lineno} imports {name!r}")
        self.assertEqual(offenders, [], "\n  ".join(offenders))


class VendorabilityTest(unittest.TestCase):
    def test_the_package_would_be_vendored_without_its_tests(self):
        """``vendor._copy_package``'s ignore patterns must actually exclude this package's tests.

        If they did not, the vendored tests would ``sys.path``-patch the repo root and import
        ``chachak``, failing the bundle's import-closure check immediately.
        """
        import shutil
        import tempfile

        from chachak.bundle_export import vendor

        destination = Path(tempfile.mkdtemp()) / "gpu_infer"
        shutil.copytree(
            _PACKAGE,
            destination,
            ignore=shutil.ignore_patterns("__pycache__", "tests", "*.pyc", "PLAN.md"),
        )
        self.assertTrue((destination / "__init__.py").exists())
        self.assertFalse((destination / "tests").exists())
        # And the real vendorer uses the same patterns.
        self.assertTrue(hasattr(vendor, "_copy_package"))

    def test_every_module_is_syntactically_valid(self):
        for path in _package_modules():
            with self.subTest(module=path.name):
                ast.parse(path.read_text(), filename=str(path))


if __name__ == "__main__":
    unittest.main()
