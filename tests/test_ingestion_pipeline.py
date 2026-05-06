from __future__ import annotations

from pathlib import Path

from rag_core.core.config import IngestionSettings, load_config
from rag_core.pipelines.ingestion import IngestionPipeline


class _DummyIndexer:
    def __init__(self) -> None:
        self.ingested_batches: list[list[dict[str, str]]] = []

    def reset_collections(self) -> None:
        pass

    def ingest_documents(self, docs):  # type: ignore[no-untyped-def]
        self.ingested_batches.append(docs)


def _build_pipeline(source_dir: Path, processed_dir: Path) -> IngestionPipeline:
    return IngestionPipeline(
        config={
            "DOCS_DIR": str(source_dir),
            "PROCESSED_DOCS_DIR": str(processed_dir),
            "DOCUMENT_STORE_PERSIST_PATH": str(processed_dir / "doc_store.pkl"),
            "TABLE_CONTEXT_WINDOW": "3",
            "SUMMARIZE_TABLES_ENABLED": "False",
        },
        ingestion_settings=IngestionSettings(),
        indexer=_DummyIndexer(),
        table_summerizer=object(),
        overwrite=False,
    )


def test_load_config_defaults_docs_dir_to_source_documents(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("DOCS_DIR", raising=False)
    config = load_config(env_path=tmp_path / ".env.does.not.exist")
    assert config["DOCS_DIR"] == "data/source_documents"


def test_iter_files_accepts_pdf_and_md_with_ignore_lists(tmp_path: Path):
    source_dir = tmp_path / "source_documents"
    processed_dir = tmp_path / "processed"
    (source_dir / "finance").mkdir(parents=True, exist_ok=True)
    (source_dir / "legal").mkdir(parents=True, exist_ok=True)
    (source_dir / "__pycache__").mkdir(parents=True, exist_ok=True)
    (source_dir / ".git").mkdir(parents=True, exist_ok=True)

    (source_dir / "finance" / "report.pdf").write_text("fake-pdf-bytes", encoding="utf-8")
    (source_dir / "finance" / "notes.md").write_text("# notes", encoding="utf-8")
    (source_dir / "legal" / "contract.txt").write_text("contract", encoding="utf-8")
    (source_dir / "README.md").write_text("guide", encoding="utf-8")
    (source_dir / "LICENSE").write_text("license", encoding="utf-8")
    (source_dir / ".gitignore").write_text("*", encoding="utf-8")
    (source_dir / "__pycache__" / "cached.md").write_text("skip", encoding="utf-8")
    (source_dir / ".git" / "config.md").write_text("skip", encoding="utf-8")

    pipeline = _build_pipeline(source_dir=source_dir, processed_dir=processed_dir)
    found_files = sorted(Path(path).name for path in pipeline._iter_files())
    assert found_files == ["notes.md", "report.pdf"]


def test_folder_name_is_used_as_tag(tmp_path: Path):
    source_dir = tmp_path / "source_documents"
    processed_dir = tmp_path / "processed"
    pipeline = _build_pipeline(source_dir=source_dir, processed_dir=processed_dir)

    assert pipeline._get_tags_for_file(str(source_dir / "finance" / "a.pdf")) == [
        "finance"
    ]
    assert pipeline._get_tags_for_file(str(source_dir / "legal" / "b.pdf")) == ["legal"]
    assert pipeline._get_tags_for_file(str(source_dir / "hr" / "c.pdf")) == ["hr"]
    assert pipeline._get_tags_for_file(str(source_dir / "root.pdf")) == ["source_documents"]


def test_run_skips_ignored_files_and_processes_pdf_and_md(tmp_path: Path, monkeypatch):
    source_dir = tmp_path / "source_documents"
    processed_dir = tmp_path / "processed"
    (source_dir / "finance").mkdir(parents=True, exist_ok=True)
    (source_dir / "legal").mkdir(parents=True, exist_ok=True)
    (source_dir / "README.md").write_text("guide", encoding="utf-8")
    (source_dir / "finance" / "demo.pdf").write_text("fake-pdf-bytes", encoding="utf-8")
    (source_dir / "legal" / "notes.md").write_text("# note", encoding="utf-8")

    pipeline = _build_pipeline(source_dir=source_dir, processed_dir=processed_dir)

    converted_paths: list[str] = []

    def _fake_convert_pdf_to_markdown(
        input_path: str, table_context_window: int, file_id: str
    ) -> tuple[str, list[dict[str, str]]]:
        del table_context_window, file_id
        converted_paths.append(input_path)
        return "# Demo", []

    monkeypatch.setattr(
        "rag_core.pipelines.ingestion.convert_pdf_to_markdown",
        _fake_convert_pdf_to_markdown,
    )
    pipeline.markdown_chunker = lambda **kwargs: [  # type: ignore[assignment]
        {"id": kwargs["file_id"], "text": kwargs["markdown"]}
    ]
    pipeline.summerize_and_chunk_table = lambda *args, **kwargs: []  # type: ignore[assignment]

    pipeline.run()

    assert len(converted_paths) == 1
    assert converted_paths[0].endswith("demo.pdf")
    assert (processed_dir / "demo.md").exists()
    assert (processed_dir / "notes.md").exists()
    assert len(pipeline.indexer.ingested_batches) == 2  # type: ignore[attr-defined]
