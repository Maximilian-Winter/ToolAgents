"""Tests for pipeline sources, sinks, chunking, and the write gate."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from ToolAgents.pipelines import (
    FileSink,
    FileSource,
    FilesSink,
    FolderSource,
    HttpSink,
    MapProcess,
    Pipeline,
    PipelineExecutionError,
    PipelineSerializationError,
    ProcessStep,
    SinkProcess,
    SourceProcess,
    SplitterConfigurationError,
    StreamSink,
    TextSource,
    build_splitter,
)


class EchoAgent:
    def get_response(self, messages, tool_registry=None, **kwargs):
        return SimpleNamespace(response="<" + messages[-1].get_as_text() + ">")


@pytest.fixture
def notes(tmp_path):
    folder = tmp_path / "notes"
    folder.mkdir()
    (folder / "a.md").write_text("alpha content", encoding="utf-8")
    (folder / "b.md").write_text("beta content", encoding="utf-8")
    (folder / "skip.txt").write_text("not markdown", encoding="utf-8")
    return folder


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def test_splitters_build_from_config():
    assert build_splitter(None) is None
    assert build_splitter("none") is not None
    assert build_splitter({"type": "simple", "chunk_size": 10}) is not None
    chunks = build_splitter(
        {"type": "recursive_character", "chunk_size": 12, "chunk_overlap": 2}
    ).get_chunks("one two three four five six")
    assert len(list(chunks)) > 1


def test_unknown_splitter_type_lists_the_known_ones():
    with pytest.raises(SplitterConfigurationError, match="Known types"):
        build_splitter({"type": "nope"})


def test_a_mistyped_splitter_option_is_reported():
    """Silently ignoring an unknown option would give the wrong chunk size."""

    with pytest.raises(SplitterConfigurationError, match="does not accept"):
        build_splitter({"type": "simple", "chunk_sixe": 10})


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def test_text_source_yields_a_string():
    process = SourceProcess(TextSource("hello"), result_key="doc")
    pipeline = Pipeline()
    pipeline.add_process(process)

    assert pipeline.run_pipeline()["inputs/doc"] == "hello"


def test_file_source_reads_one_file(tmp_path):
    target = tmp_path / "one.md"
    target.write_text("file body", encoding="utf-8")

    pipeline = Pipeline()
    pipeline.add_process(SourceProcess(FileSource(str(target)), result_key="doc"))

    assert pipeline.run_pipeline()["inputs/doc"] == "file body"


def test_folder_source_honours_the_glob_and_sorts(notes):
    pipeline = Pipeline()
    pipeline.add_process(
        SourceProcess(FolderSource(str(notes), glob="*.md"), result_key="docs")
    )

    docs = pipeline.run_pipeline()["inputs/docs"]
    assert [d["name"] for d in docs] == ["a.md", "b.md"]
    assert docs[0]["content"] == "alpha content"


def test_source_lands_in_inputs_not_outputs(notes):
    pipeline = Pipeline()
    pipeline.add_process(
        SourceProcess(FolderSource(str(notes), glob="*.md"), result_key="docs")
    )

    results = pipeline.run_pipeline()
    assert "docs" in results.inputs
    assert "docs" not in results.outputs


def test_a_source_path_can_reference_results(tmp_path):
    target = tmp_path / "dynamic.md"
    target.write_text("dynamic body", encoding="utf-8")

    pipeline = Pipeline()
    pipeline.add_process(SourceProcess(FileSource("{inputs/where}"), result_key="doc"))

    assert pipeline.run_pipeline(where=str(target))["inputs/doc"] == "dynamic body"


def test_an_unresolved_path_placeholder_is_refused():
    """A path with a hole in it would read or write the wrong place."""

    pipeline = Pipeline()
    pipeline.add_process(SourceProcess(FileSource("{inputs/missing}/x.md")))

    with pytest.raises(PipelineExecutionError, match="unresolved placeholder"):
        pipeline.run_pipeline()


def test_missing_source_file_names_itself(tmp_path):
    pipeline = Pipeline()
    pipeline.add_process(SourceProcess(FileSource(str(tmp_path / "nope.md"))))

    with pytest.raises(PipelineExecutionError, match="does not exist"):
        pipeline.run_pipeline()


def test_max_files_guards_a_too_wide_glob(notes):
    pipeline = Pipeline()
    pipeline.add_process(
        SourceProcess(FolderSource(str(notes), glob="*", max_files=1))
    )

    with pytest.raises(PipelineExecutionError, match="over the limit"):
        pipeline.run_pipeline()


def test_splitting_a_text_source_gives_a_list_of_chunks(tmp_path):
    target = tmp_path / "long.md"
    target.write_text("one two three four five six seven eight", encoding="utf-8")

    pipeline = Pipeline()
    pipeline.add_process(
        SourceProcess(
            FileSource(str(target)),
            result_key="chunks",
            splitter={"type": "simple", "chunk_size": 12},
        )
    )

    chunks = pipeline.run_pipeline()["inputs/chunks"]
    assert len(chunks) > 1
    assert all(isinstance(c, str) for c in chunks)


def test_splitting_a_record_source_keeps_the_record_fields(notes):
    pipeline = Pipeline()
    pipeline.add_process(
        SourceProcess(
            FolderSource(str(notes), glob="*.md"),
            result_key="chunks",
            splitter={"type": "simple", "chunk_size": 6},
        )
    )

    chunks = pipeline.run_pipeline()["inputs/chunks"]
    assert len(chunks) > 2
    assert {"path", "name", "content", "chunk_index"} <= set(chunks[0])
    assert chunks[0]["name"] == "a.md"


# ---------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------


def test_file_sink_writes_and_records_the_path(tmp_path):
    out = tmp_path / "nested" / "out.md"
    pipeline = Pipeline()
    pipeline.add_process(SourceProcess(TextSource("body"), result_key="doc"))
    pipeline.add_process(
        SinkProcess(FileSink(str(out)), source_key="inputs/doc", record_as="written")
    )

    results = pipeline.run_pipeline()
    assert out.read_text(encoding="utf-8") == "body"
    assert results["outputs/written"] == str(out)


def test_files_sink_writes_one_file_per_item(tmp_path, notes):
    pipeline = Pipeline()
    pipeline.add_process(
        SourceProcess(FolderSource(str(notes), glob="*.md"), result_key="docs")
    )
    pipeline.add_process(
        SinkProcess(
            FilesSink(str(tmp_path / "out" / "{name}"), content_key="content"),
            source_key="inputs/docs",
            record_as="written",
        )
    )

    written = pipeline.run_pipeline()["outputs/written"]
    assert len(written) == 2
    assert (tmp_path / "out" / "a.md").read_text(encoding="utf-8") == "alpha content"


def test_stream_sink_prints_and_is_not_gated(capsys):
    pipeline = Pipeline()
    pipeline.add_process(SourceProcess(TextSource("shown"), result_key="doc"))
    pipeline.add_process(
        SinkProcess(
            StreamSink(prefix="out: "), source_key="inputs/doc", allow_writes=False
        )
    )

    pipeline.run_pipeline()
    assert "out: shown" in capsys.readouterr().out


def test_sink_reading_a_missing_key_says_what_exists():
    pipeline = Pipeline()
    pipeline.add_process(SinkProcess(StreamSink(), source_key="outputs/nope"))

    with pytest.raises(PipelineExecutionError, match="does not exist"):
        pipeline.run_pipeline()


def test_a_sink_inside_a_map_writes_per_iteration(tmp_path, notes):
    """Composition, not new machinery: a sink nested in a map."""

    inner_sink = SinkProcess(
        FileSink(str(tmp_path / "out" / "{vars/index}.md")),
        source_key="outputs/summary",
    )
    mapper = MapProcess(
        items="docs",
        process_name="per_doc",
        agent=EchoAgent(),
        item_var="doc",
        steps=[ProcessStep("summary", "s", "sum {vars/doc}")],
    )
    mapper.add_process(inner_sink)

    pipeline = Pipeline()
    pipeline.add_process(
        SourceProcess(FolderSource(str(notes), glob="*.md"), result_key="docs")
    )
    pipeline.add_process(mapper)
    pipeline.run_pipeline()

    assert (tmp_path / "out" / "0.md").is_file()
    assert (tmp_path / "out" / "1.md").is_file()


# ---------------------------------------------------------------------------
# The write gate
# ---------------------------------------------------------------------------


def _sink_doc(path: str) -> dict:
    return {
        "schema_version": 2,
        "processes": [
            {
                "process_type": "source",
                "process_name": "load",
                "source": {"type": "text", "text": "body"},
                "result_key": "doc",
            },
            {
                "process_type": "sink",
                "process_name": "save",
                "sink": {"type": "file", "path": path},
                "from": "inputs/doc",
            },
        ],
    }


def test_a_loaded_document_cannot_write_by_default(tmp_path):
    out = tmp_path / "denied.md"
    pipeline = Pipeline.from_dict(_sink_doc(str(out)))

    with pytest.raises(PipelineExecutionError, match="allow_writes=True"):
        pipeline.run_pipeline()
    assert not out.exists()


def test_allow_writes_lets_a_loaded_document_write(tmp_path):
    out = tmp_path / "allowed.md"
    pipeline = Pipeline.from_dict(_sink_doc(str(out)), allow_writes=True)
    pipeline.run_pipeline()

    assert out.read_text(encoding="utf-8") == "body"


def test_reading_is_never_gated(tmp_path):
    """Only writing is opt-in; a source loads under the default policy."""

    target = tmp_path / "readable.md"
    target.write_text("readable", encoding="utf-8")
    document = {
        "schema_version": 2,
        "processes": [
            {
                "process_type": "source",
                "process_name": "load",
                "source": {"type": "file", "path": str(target)},
                "result_key": "doc",
            }
        ],
    }

    assert Pipeline.from_dict(document).run_pipeline()["inputs/doc"] == "readable"


def test_a_sink_built_in_python_is_permitted_by_default(tmp_path):
    """Writing the code is the intent; the gate is on loading a document."""

    out = tmp_path / "python.md"
    pipeline = Pipeline()
    pipeline.add_process(SourceProcess(TextSource("body"), result_key="doc"))
    pipeline.add_process(SinkProcess(FileSink(str(out)), source_key="inputs/doc"))
    pipeline.run_pipeline()

    assert out.is_file()


# ---------------------------------------------------------------------------
# HTTP sink
# ---------------------------------------------------------------------------


def test_http_sink_posts_json_and_reads_headers_from_env(monkeypatch):
    monkeypatch.setenv("MY_TOKEN", "sekrit")
    captured = {}

    class Response:
        status_code = 200
        text = "ok"

    def fake_request(method, url, **kwargs):
        captured.update(method=method, url=url, **kwargs)
        return Response()

    monkeypatch.setattr("requests.request", fake_request)

    sink = HttpSink(
        "https://example.test/hook",
        headers_from_env={"Authorization": "MY_TOKEN"},
    )
    pipeline = Pipeline()
    pipeline.add_process(SourceProcess(TextSource("payload"), result_key="doc"))
    pipeline.add_process(
        SinkProcess(sink, source_key="inputs/doc", record_as="status")
    )

    assert pipeline.run_pipeline()["outputs/status"] == 200
    assert captured["method"] == "POST"
    assert captured["json"] == {"content": "payload"}
    assert captured["headers"]["Authorization"] == "sekrit"


def test_http_sink_never_serializes_the_secret(monkeypatch):
    sink = HttpSink("https://example.test", headers_from_env={"Authorization": "MY_TOKEN"})
    monkeypatch.setenv("MY_TOKEN", "sekrit")

    serialized = json.dumps(sink.to_dict())
    assert "sekrit" not in serialized
    assert "MY_TOKEN" in serialized


def test_http_sink_refuses_a_literal_token():
    with pytest.raises(PipelineSerializationError, match="must not contain a literal"):
        HttpSink.from_dict({"url": "https://example.test", "token": "sekrit"})


def test_http_sink_raises_on_an_error_status(monkeypatch):
    class Response:
        status_code = 503
        text = "unavailable"

    monkeypatch.setattr("requests.request", lambda *a, **k: Response())

    pipeline = Pipeline()
    pipeline.add_process(SourceProcess(TextSource("x"), result_key="doc"))
    pipeline.add_process(
        SinkProcess(HttpSink("https://example.test"), source_key="inputs/doc")
    )

    with pytest.raises(PipelineExecutionError, match="503"):
        pipeline.run_pipeline()


def test_missing_header_env_var_names_itself(monkeypatch):
    monkeypatch.delenv("MY_TOKEN", raising=False)
    sink = HttpSink("https://example.test", headers_from_env={"Authorization": "MY_TOKEN"})

    with pytest.raises(PipelineExecutionError, match="MY_TOKEN"):
        sink.resolve_headers()


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_sources_and_sinks_round_trip(tmp_path, notes):
    document = {
        "schema_version": 2,
        "processes": [
            {
                "process_type": "source",
                "process_name": "load",
                "source": {"type": "folder", "path": str(notes), "glob": "*.md"},
                "splitter": {"type": "recursive_character", "chunk_size": 50},
                "result_key": "chunks",
            },
            {
                "process_type": "sink",
                "process_name": "save",
                "sink": {"type": "file", "path": str(tmp_path / "out.md")},
                "from": "inputs/chunks",
                "record_as": "written",
            },
        ],
    }

    once = Pipeline.from_dict(document).to_dict()
    twice = Pipeline.from_dict(once).to_dict()
    assert once == twice
    assert once["processes"][0]["source"]["glob"] == "*.md"
    assert once["processes"][1]["from"] == "inputs/chunks"


def test_unknown_source_and_sink_types_list_the_known_ones():
    with pytest.raises(PipelineSerializationError, match="Known types"):
        Pipeline.from_dict(
            {
                "schema_version": 2,
                "processes": [
                    {"process_type": "source", "source": {"type": "carrier-pigeon"}}
                ],
            }
        )
    with pytest.raises(PipelineSerializationError, match="Known types"):
        Pipeline.from_dict(
            {
                "schema_version": 2,
                "processes": [{"process_type": "sink", "sink": {"type": "smoke"}}],
            }
        )


# ---------------------------------------------------------------------------
# The shipped example
# ---------------------------------------------------------------------------

EXAMPLE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "examples", "agents", "pipeline",
)
EXAMPLE_JSON = os.path.join(EXAMPLE_DIR, "folder_to_file_pipeline.json")


class PointAgent:
    def get_response(self, messages, tool_registry=None, **kwargs):
        text = messages[-1].get_as_text()
        return SimpleNamespace(response="POINT: " + text.splitlines()[-1][:40])


@pytest.mark.skipif(
    not os.path.exists(EXAMPLE_JSON), reason="example pipeline not present"
)
def test_shipped_folder_to_file_example_runs(tmp_path):
    with open(EXAMPLE_JSON, encoding="utf-8") as handle:
        document = json.load(handle)

    pipeline = Pipeline.from_dict(
        document, build_agents=False, default_agent=PointAgent(), allow_writes=True
    )
    results = pipeline.run_pipeline(
        notes_dir=os.path.join(EXAMPLE_DIR, "notes"),
        out_dir=str(tmp_path),
        title="Otters, briefly",
    )

    assert len(results["inputs/chunks"]) >= 2
    assert len(results["outputs/points"]) == len(results["inputs/chunks"])
    assert (tmp_path / "digest.md").is_file()


@pytest.mark.skipif(
    not os.path.exists(EXAMPLE_JSON), reason="example pipeline not present"
)
def test_shipped_example_refuses_to_write_by_default(tmp_path):
    with open(EXAMPLE_JSON, encoding="utf-8") as handle:
        document = json.load(handle)

    pipeline = Pipeline.from_dict(
        document, build_agents=False, default_agent=PointAgent()
    )
    with pytest.raises(PipelineExecutionError, match="allow_writes=True"):
        pipeline.run_pipeline(
            notes_dir=os.path.join(EXAMPLE_DIR, "notes"),
            out_dir=str(tmp_path),
            title="T",
        )
    assert not (tmp_path / "digest.md").exists()
