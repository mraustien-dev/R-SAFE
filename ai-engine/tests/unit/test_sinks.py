import json

import pytest
import requests

from rsafe.config import HttpSinkConfig, SinkConfig
from rsafe.sinks.file_sink import FileSink
from rsafe.sinks.http_sink import HttpSink, make_sink


class FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


@pytest.fixture
def violation_folder(tmp_path):
    folder = tmp_path / "violations" / "2026-11-06" / "V-20261106-143212-0007"
    folder.mkdir(parents=True)
    (folder / "violation.json").write_text(json.dumps({"violation_id": "x"}), encoding="utf-8")
    (folder / "frame.jpg").write_bytes(b"frame")
    (folder / "rider.jpg").write_bytes(b"rider")
    return folder


def refuse(*args, **kwargs):
    raise requests.ConnectionError("connection refused")


def test_http_sink_failure_queues_folder(tmp_path, violation_folder, monkeypatch):
    monkeypatch.setattr(requests, "post", refuse)
    pending = tmp_path / "pending_upload.txt"

    HttpSink("http://backend/api/violations", 10, pending).send(violation_folder)

    assert pending.read_text(encoding="utf-8").splitlines() == [str(violation_folder)]
    assert (violation_folder / "violation.json").exists()  # evidence kept


def test_http_sink_non_2xx_queues_folder(tmp_path, violation_folder, monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(500))
    pending = tmp_path / "pending_upload.txt"

    HttpSink("http://backend/api/violations", 10, pending).send(violation_folder)

    assert pending.read_text(encoding="utf-8").splitlines() == [str(violation_folder)]


def test_http_sink_posts_record_and_images(tmp_path, violation_folder, monkeypatch):
    calls = []

    def fake_post(url, files, timeout):
        calls.append((url, {name: value[1] for name, value in files.items()}, timeout))
        return FakeResponse(201)

    monkeypatch.setattr(requests, "post", fake_post)
    pending = tmp_path / "pending_upload.txt"

    HttpSink("http://backend/api/violations", 10, pending).send(violation_folder)

    url, files, timeout = calls[0]
    assert url == "http://backend/api/violations" and timeout == 10
    assert set(files) == {"record", "frame", "rider"}
    assert json.loads(files["record"]) == {"violation_id": "x"}
    assert not pending.exists()


def test_http_sink_retry_clears_on_success(tmp_path, violation_folder, monkeypatch):
    pending = tmp_path / "pending_upload.txt"
    sink = HttpSink("http://backend/api/violations", 10, pending)
    monkeypatch.setattr(requests, "post", refuse)
    sink.send(violation_folder)

    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(201))

    assert sink.retry_pending() == 1
    assert pending.read_text(encoding="utf-8") == ""


def test_http_sink_retry_keeps_still_failing(tmp_path, violation_folder, monkeypatch):
    pending = tmp_path / "pending_upload.txt"
    sink = HttpSink("http://backend/api/violations", 10, pending)
    monkeypatch.setattr(requests, "post", refuse)
    sink.send(violation_folder)

    assert sink.retry_pending() == 0
    assert pending.read_text(encoding="utf-8").splitlines() == [str(violation_folder)]


def test_make_sink_selects_kind(tmp_path):
    output_dir = tmp_path / "output" / "violations"

    assert isinstance(make_sink(SinkConfig("file", HttpSinkConfig("", 10)), output_dir), FileSink)
    http = make_sink(SinkConfig("http", HttpSinkConfig("http://b/api", 5)), output_dir)
    assert isinstance(http, HttpSink)
    assert http.pending_file == tmp_path / "output" / "pending_upload.txt"
