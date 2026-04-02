import ast
import collections
import unittest

from _api_symbol_manifest import SURFACE_MANIFEST
from _workspace_bootstrap import REPO_ROOT
from _suite_manifest import EXCLUDED_TESTS, MPI_TESTS, SERIAL_TESTS


def parse_class_symbols(path, class_name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    raise AssertionError(f"Class {class_name} not found in {path.relative_to(REPO_ROOT)}")


class TestCleanupGuards(unittest.TestCase):
    def test_no_duplicate_python_class_methods(self):
        package_root = REPO_ROOT / "vbcsr"

        duplicate_methods = []
        for path in package_root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:
                if not isinstance(node, ast.ClassDef):
                    continue
                method_names = [
                    child.name
                    for child in node.body
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                duplicates = [
                    name for name, count in collections.Counter(method_names).items() if count > 1
                ]
                for name in duplicates:
                    duplicate_methods.append(f"{path.relative_to(REPO_ROOT)}::{node.name}.{name}")

        self.assertFalse(duplicate_methods, "\n".join(duplicate_methods))

    def test_docs_match_current_cleanup_contracts(self):
        api_reference = (REPO_ROOT / "doc" / "api_reference.md").read_text(encoding="utf-8")
        completeness_matrix = (REPO_ROOT / "doc" / "completeness_matrix.md").read_text(encoding="utf-8")
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("from_scipy(cls, spmat: Any, comm=None, root: int = 0) -> 'VBCSR'", api_reference)
        self.assertNotIn("from_scipy(cls, spmat: Any, comm=None) -> 'VBCSR'", api_reference)
        self.assertIn("Scalar and slicing indexing are currently unsupported.", api_reference)
        self.assertNotIn("Access individual elements", api_reference)
        self.assertIn("## AtomicData", api_reference)
        self.assertIn("## ImageContainer", api_reference)
        self.assertIn("## DistGraph", api_reference)
        self.assertIn("`atom_types` remains the internal type-id surface", api_reference)
        self.assertIn("`atomic_numbers` / `z` expose true atomic numbers", api_reference)
        self.assertIn("# Repo Completeness Matrix", completeness_matrix)
        self.assertIn("Atomic topology", completeness_matrix)
        self.assertIn("Native regression orchestration", completeness_matrix)
        self.assertIn("tests/run_python_suite.py", readme)
        self.assertIn("vbcsr/core/test/run_native_suite.py", readme)
        self.assertNotIn("run_all_tests.py", readme)
        self.assertNotIn("run_cmake_registered_tests.py", readme)

    def test_every_python_test_file_is_in_the_manifest(self):
        manifest = set(SERIAL_TESTS) | set(MPI_TESTS) | set(EXCLUDED_TESTS)
        test_files = {path.name for path in (REPO_ROOT / "tests").glob("test_*.py")}
        self.assertEqual(manifest, test_files)

    def test_native_suite_has_atomic_runner(self):
        native_suite = (REPO_ROOT / "vbcsr" / "core" / "test" / "run_native_suite.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("run_all_tests.py", native_suite)
        self.assertIn("run_cmake_registered_tests.py", native_suite)

    def test_surface_manifest_matches_code_docs_and_tests(self):
        package_init = (REPO_ROOT / "vbcsr" / "__init__.py").read_text(encoding="utf-8")
        api_reference = (REPO_ROOT / "doc" / "api_reference.md").read_text(encoding="utf-8")
        wrapper_symbols = {}

        for surface_name, surface in SURFACE_MANIFEST.items():
            self.assertIn(surface.package_export, package_init, surface_name)
            self.assertTrue((REPO_ROOT / surface.pybind_file).is_file(), surface.pybind_file)
            self.assertTrue(surface.docs_checks, surface_name)
            for token in surface.docs_checks:
                self.assertIn(token, api_reference, f"{surface_name}: missing docs token {token!r}")

            if surface.wrapper_file and surface.wrapper_class:
                path = REPO_ROOT / surface.wrapper_file
                self.assertTrue(path.is_file(), surface.wrapper_file)
                wrapper_symbols[surface_name] = parse_class_symbols(path, surface.wrapper_class)
            else:
                wrapper_symbols[surface_name] = set()

        maintained_tests = set(SERIAL_TESTS) | set(MPI_TESTS) | set(EXCLUDED_TESTS)

        for surface_name, surface in SURFACE_MANIFEST.items():
            pybind_text = (REPO_ROOT / surface.pybind_file).read_text(encoding="utf-8")
            symbols = wrapper_symbols[surface_name]
            seen_names = set()

            for symbol in surface.symbols:
                for name in (symbol.canonical,) + symbol.aliases:
                    self.assertNotIn(name, seen_names, f"{surface_name}: duplicate manifest name {name}")
                    seen_names.add(name)

                for token in symbol.pybind_checks:
                    self.assertIn(token, pybind_text, f"{surface_name}.{symbol.canonical}: missing pybind token {token!r}")

                for name in symbol.python_checks:
                    self.assertIn(name, symbols, f"{surface_name}.{symbol.canonical}: missing Python symbol {name}")

                for token in symbol.docs_checks:
                    self.assertIn(token, api_reference, f"{surface_name}.{symbol.canonical}: missing docs token {token!r}")

                self.assertTrue(
                    symbol.tests,
                    f"{surface_name}.{symbol.canonical}: every symbol must name at least one maintained test owner",
                )
                for test_path in symbol.tests:
                    full_path = REPO_ROOT / test_path
                    self.assertTrue(full_path.exists(), f"{surface_name}.{symbol.canonical}: missing test path {test_path}")
                    if full_path.parent == REPO_ROOT / "tests" and full_path.name.startswith("test_"):
                        self.assertIn(
                            full_path.name,
                            maintained_tests,
                            f"{surface_name}.{symbol.canonical}: {full_path.name} is not in the maintained Python suite manifest",
                        )


if __name__ == "__main__":
    unittest.main()
