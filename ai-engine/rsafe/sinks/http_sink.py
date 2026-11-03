"""Send violation folders to the backend; queue them on disk when it is unreachable."""

import logging
from pathlib import Path

import requests

from rsafe.config import SinkConfig
from rsafe.sinks.base import Sink
from rsafe.sinks.file_sink import FileSink

logger = logging.getLogger("rsafe.sinks.http")

IMAGE_FIELDS = ("frame", "rider", "plate")


class HttpSink:
    def __init__(self, url: str, timeout_s: float, pending_file: Path):
        self.url = url
        self.timeout_s = timeout_s
        self.pending_file = Path(pending_file)

    def _post(self, folder: Path) -> bool:
        files = {
            "record": (
                "violation.json",
                (folder / "violation.json").read_text(encoding="utf-8"),
                "application/json",
            )
        }
        for name in IMAGE_FIELDS:
            image = folder / f"{name}.jpg"
            if image.exists():
                files[name] = (image.name, image.read_bytes(), "image/jpeg")
        try:
            response = requests.post(self.url, files=files, timeout=self.timeout_s)
        except requests.RequestException as error:
            logger.warning("[R-SAFE] Upload failed for %s: %s", folder.name, error)
            return False
        if not 200 <= response.status_code < 300:
            logger.warning(
                "[R-SAFE] Upload rejected for %s: HTTP %s", folder.name, response.status_code
            )
            return False
        return True

    def send(self, folder: Path) -> None:
        if not self._post(folder):
            self.pending_file.parent.mkdir(parents=True, exist_ok=True)
            with self.pending_file.open("a", encoding="utf-8") as pending:
                pending.write(f"{folder}\n")

    def retry_pending(self) -> int:
        if not self.pending_file.exists():
            return 0
        folders = [
            Path(line) for line in self.pending_file.read_text(encoding="utf-8").splitlines() if line
        ]
        still_failing = [folder for folder in folders if not self._post(folder)]
        self.pending_file.write_text(
            "".join(f"{folder}\n" for folder in still_failing), encoding="utf-8"
        )
        return len(folders) - len(still_failing)


def make_sink(cfg: SinkConfig, output_dir: Path) -> Sink:
    if cfg.kind == "http":
        return HttpSink(cfg.http.url, cfg.http.timeout_s, Path(output_dir).parent / "pending_upload.txt")
    return FileSink()
