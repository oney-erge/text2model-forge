import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Every place that installs the build toolchain outside of pip's build isolation.
# Dependabot only bumps pyproject.toml, so a stale copy here would keep CI and the
# security audit on the old setuptools.
PINNED_SITES = (
    ".github/workflows/ci.yml",
    ".github/workflows/release.yml",
    ".github/workflows/security.yml",
    "Dockerfile",
    "text2model-forge.ps1",
    "text2model-forge.sh",
)


def _build_setuptools_version() -> str:
    requires = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["build-system"]["requires"]
    (pin,) = [item for item in requires if item.startswith("setuptools==")]
    return pin.removeprefix("setuptools==")


def test_setuptools_pins_match_build_system() -> None:
    expected = _build_setuptools_version()
    pattern = re.compile(r'setuptools==([0-9.]+)|SETUPTOOLS_VERSION="([0-9.]+)"|\$SetuptoolsVersion = "([0-9.]+)"')
    for relative in PINNED_SITES:
        text = (ROOT / relative).read_text(encoding="utf-8")
        found = {next(group for group in match.groups() if group) for match in pattern.finditer(text)}
        assert found == {expected}, f"{relative} pins setuptools {sorted(found)}, pyproject.toml pins {expected}"
