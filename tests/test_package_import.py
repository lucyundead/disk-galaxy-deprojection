import tomllib
from pathlib import Path


def test_package_imports():
    import dgdp

    # keep __init__.__version__ in lockstep with pyproject (the release workflow gates tag ==
    # pyproject; this closes the remaining leg so a bump can't miss one of the two)
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert dgdp.__version__ == pyproject["project"]["version"]
