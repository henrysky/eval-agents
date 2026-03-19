"""Bloomberg financial news dataset loader.

This module provides classes for loading and accessing the Bloomberg financial news dataset.
"""

import logging
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class BloombergNewsExample(BaseModel):
    """A single Bloomberg financial news article."""

    example_id: int = Field(description="Unique identifier for the article.")
    title: str = Field(description="Headline of the news article.")
    content: str = Field(description="Full text of the article.")
    date: str | None = Field(default=None, description="Publication date.")
    source: str | None = Field(default=None, description="Source of the article.")


class BloombergFinancialNewsDataset:
    """Loader and manager for the Bloomberg Financial News dataset."""

    def __init__(self) -> None:
        self._df: pd.DataFrame | None = None
        self._examples: list[BloombergNewsExample] | None = None

    def _load_data(self) -> None:
        """Load dataset from Hugging Face."""
        if self._df is not None:
            return

        logger.info("Loading Bloomberg Financial News dataset...")

        ds = load_dataset("danidanou/Bloomberg_Financial_News")

        # Convert to pandas (assumes 'train' split exists)
        self._df = ds["train"].to_pandas()

        logger.info(f"Loaded {len(self._df)} articles")

        # Normalize column names (adjust if dataset schema differs)
        # Common columns might include: 'title', 'content', 'date'
        required_cols = set(self._df.columns)

        self._examples = []
        for idx, row in self._df.iterrows():
            self._examples.append(
                BloombergNewsExample(
                    example_id=idx,
                    title=row.get("Headline", ""),
                    content=row.get("Article", ""),
                    date=row.get("Date").strftime("%Y-%m-%d %H:%M:%S"),
                    source=row.get("Link"),
                )
            )

    @property
    def dataframe(self) -> pd.DataFrame:
        """Return dataset as pandas DataFrame."""
        self._load_data()
        assert self._df is not None
        return self._df

    @property
    def examples(self) -> list[BloombergNewsExample]:
        """Return dataset as structured examples."""
        self._load_data()
        assert self._examples is not None
        return self._examples

    def __len__(self) -> int:
        self._load_data()
        assert self._examples is not None
        return len(self._examples)

    def __getitem__(self, index: int) -> BloombergNewsExample:
        self._load_data()
        assert self._examples is not None
        return self._examples[index]

    def sample(self, n: int = 10, random_state: int | None = None) -> list[BloombergNewsExample]:
        """Sample random articles."""
        sampled_df = self.dataframe.sample(n=min(n, len(self)), random_state=random_state)

        return [
            BloombergNewsExample(
                example_id=idx,
                title=row.get("Headline", ""),
                content=row.get("Article", ""),
                date=row.get("Date").strftime("%Y-%m-%d %H:%M:%S"),
                source=row.get("Link"),
            )
            for idx, row in sampled_df.iterrows()
        ]
