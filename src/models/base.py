"""Common model interface and prediction schema."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Prediction:
    pair_id: str
    model: str
    model_revision: str
    audio: str
    question: str
    prediction: str
    ground_truth: str
    condition: str
    intervention_role: str | None
    seed: int
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AudioLanguageModel(ABC):
    model_id: str
    revision: str

    @abstractmethod
    def generate(self, audio: str | Path, question: str) -> str:
        """Generate one textual answer from an audio file and question."""

