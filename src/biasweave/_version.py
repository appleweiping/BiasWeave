"""Installed distribution version exposed to the runtime."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("biasweave")
except PackageNotFoundError:  # pragma: no cover - only an uninstalled source checkout
    __version__ = "0+unknown"
