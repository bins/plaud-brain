from pathlib import Path

from plaud_brain.adapters.downloader import DownloadResult


def test_download_result_shape(tmp_path: Path):
    p = tmp_path / "x.audio"
    p.write_bytes(b"abc")
    r = DownloadResult(path=p, sha256="x", bytes_written=3)
    assert r.path.exists()
    assert r.bytes_written == 3
