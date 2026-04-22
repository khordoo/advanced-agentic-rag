import logging
import os
from pathlib import Path

from mcp.server.auth import settings

from rag_core.converter import convert_pdf_to_markdown
from rag_core.chunker import chunk_markdown
from rag_core.core.config import load_config, setup_logging
from rag_core.indexing.base import BaseIndexer
from rag_core.indexing.qdrant_indexer import QdrantIndexer
from rag_core.utils import Stopwatch, generate_file_id
from rag_core.agents import TableSummarizer
from rag_core.chunker import summerize_and_chunk_table
from rag_core.core.config import IngestionSettings

setup_logging()
logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Orchestrates the end-to-end data ingestion process for the RAG system.

    This pipeline handles the conversion of raw PDFs to Markdown (including enriched tables),
    hierarchical chunking of the content, and persistence of documents and embeddings
    into their respective stores.
    """

    _IGNORE_DIR = {".DS_Store"}

    def __init__(
        self,
        config: dict[str, str],
        ingestion_settings: IngestionSettings | None = None,
        indexer: BaseIndexer | None = None,
        table_summerizer: TableSummarizer | None = None,
        overwrite: bool = True,
    ):
        """
        Initializes the ingestion pipeline with configuration and storage backends.

        Args:
            config: Configuration dictionary containing paths and parameters.
            vector_store: Vector database for searchable embeddings. Defaults to Chroma.
            doc_store: Document store for full parent chunks. Defaults to a dict-based store.
            overwrite: If True, resets the vector store collection before processing.
        """
        self.config = config
        self.ingestion_settings = ingestion_settings or IngestionSettings()
        self.indexer = indexer or QdrantIndexer()
        self.table_summarizer = table_summerizer or TableSummarizer()
        self.markdown_chunker = chunk_markdown
        self.summerize_and_chunk_table = summerize_and_chunk_table
        self.ingestion_dir = self.config.get("DOCS_DIR", "docs")
        self.processed_dir = self.config.get("PROCESSED_DOCS_DIR", "data/processed")
        self.store_persistence_path = self.config.get("DOCUMENT_STORE_PERSIST_PATH")
        self.table_context_window = int(self.config.get("TABLE_CONTEXT_WINDOW", 3))
        self.summarize_tables_enabled: bool = bool(
            self.config.get("SUMMARIZE_TABLES_ENABLED", True)
        )
        logger.info(
            "IngestionPipeline initialized",
            extra={
                "ingestion_dir": self.ingestion_dir,
                "processed_dir": self.processed_dir,
                "summarize_tables": self.summarize_tables_enabled,
            },
        )
        if overwrite:
            self.indexer.reset_collections()
            logger.info("Vector and document DB collection reset")

    def run(self):
        """
        Executes the ingestion loop over all files in the source directory.

        For each file, it:
        1. Converts PDF to Markdown with enriched table context.
        2. Chunks the Markdown into hierarchical structures.
        3. Persists results into the vector and document stores.
        """
        # TODO: having a failure list and proper handling of failed files
        logger.info("Starting ingestion run")
        failed_files = []
        total_processed = 0
        for file_path in self._iter_files():
            try:
                # Get tags dynamically per file (even if it's currently hardcoded)
                current_tags = self._get_tags_for_file(file_path)
                current_scopes = self._get_scopes_for_file(file_path)

                logger.info("Processing file: %s", file_path)

                # Generate a unique, sanitised file identifier
                file_id = generate_file_id(file_path, self.ingestion_dir)
                logger.info(
                    "Generated file identity",
                    extra={"file_id": file_id, "original": file_path},
                )

                with Stopwatch("Markdown conversion"):
                    markdown, tables = convert_pdf_to_markdown(
                        input_path=file_path,
                        table_context_window=self.table_context_window,
                        file_id=file_id,
                    )
                logger.info(
                    "Conversion complete",
                    extra={"markdown_chars": len(markdown), "table_count": len(tables)},
                )

                table_chunks = self.summerize_and_chunk_table(
                    tables,
                    tags=current_tags,
                    scopes=current_scopes,
                    model=self.table_summarizer,
                    summerize_enabled=self.summarize_tables_enabled,
                )
                logger.info(
                    "Table summarization complete",
                    extra={"table_chunks": len(table_chunks)},
                )

                chunks = self.markdown_chunker(
                    markdown=markdown,
                    source_path=file_path,
                    tags=current_tags,
                    scopes=current_scopes,
                    config=self.config,
                    file_id=file_id,
                )
                logger.info(
                    "Chunked markdown into parent chunks",
                    extra={"parent_chunks": len(chunks)},
                )

                md_file_name = Path(self.processed_dir) / f"{Path(file_path).stem}.md"
                md_file_name.parent.mkdir(parents=True, exist_ok=True)
                md_file_name.write_text(markdown, encoding="utf-8")
                logger.info(
                    "Markdown snapshot stored", extra={"path": str(md_file_name)}
                )

                total_chunks_before_extend = len(chunks)
                chunks.extend(table_chunks)
                logger.info(
                    "Extended chunks with table entries",
                    extra={
                        "doc_chunks": total_chunks_before_extend,
                        "table_chunks": len(table_chunks),
                        "total_chunks": len(chunks),
                    },
                )

                self.indexer.ingest_documents(chunks)
                total_processed += 1
            except Exception as e:
                failed_files.append(file_path)
                logger.warning(f"Failed to ingest: {e}", extra={"file_path": file_path})

        logger.info(
            f"Ingestion run completed: Failed files: {len(failed_files)}, Successful files: {total_processed} "
        )

    def _get_tags_for_file(self, file_path: str) -> list[str]:
        """
        PLACEHOLDER: Extracts or assigns tags for a specific file.
        TODO: Refactor this to use custom patterns, or LLM classification. Update as needed
        Currently assigns logical tags based on the file's directory structure.

        Currently returns the immediate parent folder name as a tag.
        Future iterations may include deeper path analysis or LLM-based categorization.

        Args:
            file_path: The absolute path to the file being processed.

        Returns:
            A list of string tags (e.g., ['legal']).
        """
        path_obj = Path(file_path)
        parent_folder = path_obj.parent.name

        # Handle the case where .parent.name might return empty or a dot
        if not parent_folder or parent_folder == ".":
            # Fallback to the actual name of the ingestion directory defined in config
            parent_folder = Path(self.ingestion_dir).name

        # Return as a list with consistent lowercasing
        return [parent_folder.lower()]

    def _get_scopes_for_file(self, file_path: str) -> list[str]:
        """
        Assigns security/access control scopes for a file.

        PLACEHOLDER: Currently defaults to ['public'].
        TODO: Implement logic to assign 'admin', 'hr', or 'restricted'
        based on file metadata or source location.
        """
        return ["public"]

    def _iter_files(self):
        """
        Yields file paths from the ingestion directory, skipping ignored items and hidden files.

        Yields:
            str: Absolute path to the file.
        """
        for root, _, files in os.walk(self.ingestion_dir):
            for file in files:
                # Skip hidden/system files (like .gitkeep or .DS_Store)
                if file.startswith("."):
                    continue

                # Skip explicitly ignored files/dirs
                if file in self._IGNORE_DIR:
                    continue

                # Only yield files that match our supported formats
                path_obj = Path(file)
                if (
                    path_obj.suffix.lower()
                    in self.ingestion_settings.ALLOWED_EXTENSIONS
                ):
                    yield os.path.join(root, file)
                else:
                    logger.debug("Skipping unsupported file type: %s", file)
        logger.info(
            f"Completed iterating through all files in folder: {self.ingestion_dir}"
        )


if __name__ == "__main__":
    cfg = load_config()
    p = IngestionPipeline(config=cfg, overwrite=True)
    p.run()
