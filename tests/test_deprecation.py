"""Tests for the deprecation warnings of verbatim_core and verbatim_rag."""

import importlib
import inspect
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import pytest
from verbatim_core import VerbatimDeprecationWarning
from verbatim_core._deprecation import REMOVED_IN, deprecated, warn_deprecated_argument

OLD_EXTRACT_MESSAGE = (
    f"old_extract is deprecated since 0.3.0 and will be removed in {REMOVED_IN}. Use extract()."
)

_UNSET = object()


def build_extractor(model=_UNSET):
    """Stands in for a library function with a deprecated argument."""
    if model is not _UNSET:
        warn_deprecated_argument("model", use="llm_client")
    return "extractor"


def test_is_a_deprecation_warning():
    assert issubclass(VerbatimDeprecationWarning, DeprecationWarning)


class TestDeprecatedFunction:
    def test_warns_with_the_library_category_and_message_when_called(self):
        @deprecated("old_extract", use="extract()")
        def old_extract():
            return "spans"

        with pytest.warns(VerbatimDeprecationWarning) as record:
            assert old_extract() == "spans"
        assert [str(w.message) for w in record] == [OLD_EXTRACT_MESSAGE]

    def test_decorating_does_not_warn(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            @deprecated("old_extract", use="extract()")
            def old_extract():
                return "spans"

        assert caught == []

    def test_is_marked_for_type_checkers(self):
        @deprecated("old_extract", use="extract()")
        def old_extract():
            return "spans"

        assert old_extract.__deprecated__ == OLD_EXTRACT_MESSAGE


class TestDeprecatedClass:
    def test_warns_when_used_not_when_its_module_is_imported(self, tmp_path, monkeypatch):
        (tmp_path / "legacy_extractors.py").write_text(
            "from verbatim_core._deprecation import deprecated\n"
            "\n"
            "\n"
            '@deprecated("LegacyExtractor", use="NewExtractor")\n'
            "class LegacyExtractor:\n"
            "    pass\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.delitem(sys.modules, "legacy_extractors", raising=False)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            module = importlib.import_module("legacy_extractors")
        assert not [w for w in caught if issubclass(w.category, VerbatimDeprecationWarning)]

        with pytest.warns(VerbatimDeprecationWarning, match="^LegacyExtractor is deprecated"):
            module.LegacyExtractor()


class TestDeprecatedArgument:
    def test_does_not_warn_when_the_argument_is_left_out(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert build_extractor() == "extractor"
        assert caught == []

    def test_warns_at_the_line_that_passed_it(self):
        with pytest.warns(VerbatimDeprecationWarning) as record:
            call_line = inspect.currentframe().f_lineno + 1
            build_extractor(model="gpt-4o-mini")

        [warning] = record
        assert str(warning.message) == (
            f"model is deprecated since 0.3.0 and will be removed in {REMOVED_IN}. Use llm_client."
        )
        assert (warning.filename, warning.lineno) == (__file__, call_line)


REPO_ROOT = Path(__file__).resolve().parents[1]
MISSING_DEPENDENCY = 77


def run_in_fresh_interpreter(script: str) -> subprocess.CompletedProcess:
    """Run a script from the repository root, skipping when a third-party dependency is missing."""
    guarded = (
        "try:\n"
        + textwrap.indent(textwrap.dedent(script), "    ")
        + f"\nexcept ModuleNotFoundError:\n    raise SystemExit({MISSING_DEPENDENCY})\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", guarded], cwd=REPO_ROOT, capture_output=True, text=True
    )
    if result.returncode == MISSING_DEPENDENCY:
        pytest.skip("the package's dependencies are not installed")
    return result


@pytest.mark.parametrize("package", ["verbatim_core", "verbatim_rag"])
def test_importing_in_a_fresh_interpreter_emits_no_deprecation_warning(package):
    result = run_in_fresh_interpreter(
        f"""
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import {package}
        from verbatim_core import VerbatimDeprecationWarning
        ours = [w for w in caught if issubclass(w.category, VerbatimDeprecationWarning)]
        assert not ours, [str(w.message) for w in ours]
        """
    )
    assert result.returncode == 0, result.stderr


def test_verbatim_rag_exports_the_same_warning_class():
    result = run_in_fresh_interpreter(
        """
        import verbatim_core
        import verbatim_rag
        assert verbatim_rag.VerbatimDeprecationWarning is verbatim_core.VerbatimDeprecationWarning
        """
    )
    assert result.returncode == 0, result.stderr
