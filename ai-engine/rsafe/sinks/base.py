"""Where finished violation folders go after they are written."""

from pathlib import Path
from typing import Protocol


class Sink(Protocol):
    def send(self, folder: Path) -> None: ...
