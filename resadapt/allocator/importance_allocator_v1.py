"""Compatibility shim: implementation is in ``importance_allocator_v2``.

Prefer ``from resadapt.allocator.importance_allocator_v2 import ...``.
"""

from resadapt.allocator.importance_allocator_v2 import (
    ContrastiveDifferentiator,
    CrossModalMatcher,
    DifferentiableImportanceAllocator,
    DualPathEncoder,
    FrameInformationEncoder,
    SparseGatingModule,
    exists,
)

__all__ = [
    "ContrastiveDifferentiator",
    "CrossModalMatcher",
    "DifferentiableImportanceAllocator",
    "DualPathEncoder",
    "FrameInformationEncoder",
    "SparseGatingModule",
    "exists",
]
