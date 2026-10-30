"""Overlay RoboTwin OpenPI extensions while retaining upstream modules."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
