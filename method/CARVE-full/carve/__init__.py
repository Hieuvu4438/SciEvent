"""CARVE — Clause-level Argument Recovery Via sEgmentation.

A span-segmentation model for scientific event argument extraction on the
SciEvent benchmark (Dong et al., EMNLP 2025).

The package is organised as:

    carve.paths      resolution of the read-only SciEvent benchmark paths
    carve.data       data contract, label inventories, BIO encode/decode
    carve.model      encoder + ROLE / AAO / TYPE heads
    carve.crf        optional linear-chain CRF (ablation; reported as negative)
    carve.train      training loop, checkpoint selection, prediction export
    carve.decode     posterior caching, calibrated decoding, frozen evaluation
    carve.evaluate   thin wrapper over the benchmark's OWN scorer
    carve.analysis   error taxonomy and per-role / domain / type / length tables
"""

__version__ = "1.0.0"

from carve.data import AAO_TYPES, EVENT_TYPES, ROLE_TYPES, Window, load_split
from carve.model import CarveModel

__all__ = [
    "__version__",
    "AAO_TYPES",
    "EVENT_TYPES",
    "ROLE_TYPES",
    "Window",
    "load_split",
    "CarveModel",
]
