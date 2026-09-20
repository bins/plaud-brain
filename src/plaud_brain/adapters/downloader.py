from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    sha256: str
    bytes_written: int


class AudioDownloader:
    def __init__(self, timeout_seconds: int = 120):
        self.timeout_seconds = timeout_seconds

    def download(self, url: str, destination: Path) -> DownloadResult:
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_suffix(destination.suffix + ".part")
        digest = hashlib.sha256()
        total = 0
        request = Request(url, headers={"User-Agent": "plaud-brain/0.1"})
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response, tmp.open("wb") as fh:
                while chunk := response.read(1024 * 1024):
                    fh.write(chunk)
                    digest.update(chunk)
                    total += len(chunk)
                fh.flush()
                os.fsync(fh.fileno())
            if total == 0:
                raise RuntimeError("download produced an empty file")
            tmp.replace(destination)
            return DownloadResult(destination, digest.hexdigest(), total)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
