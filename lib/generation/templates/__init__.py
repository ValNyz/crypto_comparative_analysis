# =============================================================================
# FILE: lib/generation/templates/__init__.py
# =============================================================================
"""Strategy templates module."""

from .base import INDICATORS_BLOCK, REGIME_DETECTION_BLOCK
from .standard import STRATEGY_TEMPLATE_STANDARD
from .funding import STRATEGY_TEMPLATE_FUNDING
from .external_block import EXTERNAL_LOADERS_BLOCK

__all__ = [
    "INDICATORS_BLOCK",
    "REGIME_DETECTION_BLOCK",
    "STRATEGY_TEMPLATE_STANDARD",
    "STRATEGY_TEMPLATE_FUNDING",
    "EXTERNAL_LOADERS_BLOCK",
]
