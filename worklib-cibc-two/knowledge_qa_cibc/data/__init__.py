"""Data loading and management for knowledge QA evaluation.

This module provides tools for loading and managing benchmark datasets
like DeepSearchQA.
"""

from .bloombergfinance import BloombergNewsExample, BloombergFinancialNewsDataset


__all__ = [
    "BloombergNewsExample",
    "BloombergFinancialNewsDataset",
]
