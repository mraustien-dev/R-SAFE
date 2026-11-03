"""Default sink: the evidence folder on disk is the hand-off."""

import logging
from pathlib import Path

logger = logging.getLogger("rsafe.sinks.file")


class FileSink:
    def send(self, folder: Path) -> None:
        logger.debug("[R-SAFE] Evidence kept on disk: %s", folder)
