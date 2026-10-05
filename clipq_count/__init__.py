"""CLIPQ-Count: query-guided vision-language few-shot object counting (IEEE INDICON 2025)."""
from .model import CLIPCountingModel
from .loss import CombinedCountingLoss
from .data import FSC147Dataset, collate_fn, verify_dataset

__all__ = ["CLIPCountingModel", "CombinedCountingLoss", "FSC147Dataset", "collate_fn", "verify_dataset"]
__version__ = "1.0.0"
