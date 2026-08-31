from __future__ import annotations

from importlib.metadata import version

import pytest

from biasweave import __version__
from biasweave.cli import main


def test_runtime_distribution_and_cli_versions_agree(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert __version__ == version("biasweave") == "0.2.0"
    with pytest.raises(SystemExit) as raised:
        main(["--version"])
    assert raised.value.code == 0
    assert capsys.readouterr().out == f"BiasWeave {__version__}\n"
