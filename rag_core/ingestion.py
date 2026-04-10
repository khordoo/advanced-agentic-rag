import logging
import os
import json
from pathlib import Path
from rag_core.converter import convert_pdf_to_markdown
from rag_core.chunker import chunk_markdown, store_chunks
from rag_core.store import VectorStore, DocumentStore
from rag_core.config import load_config, setup_logging
from rag_core.utils import Stopwatch
from rag_core.agents import TableSummarizer
from rag_core.chunker import summerize_and_chunk_table
setup_logging()
logger = logging.getLogger(__name__)

class IngestionPipeline:
    """
    Orchestrates the end-to-end data ingestion process for the RAG system.

    This pipeline handles the conversion of raw PDFs to Markdown (including enriched tables),
    hierarchical chunking of the content, and persistence of documents and embeddings
    into their respective stores.
    """
    _IGNORE_DIR = {'.DS_Store'}

    def __init__(
        self,
        config: dict[str, str],
        vector_store: VectorStore | None = None,
        doc_store: DocumentStore | None = None,
        table_summerizer: TableSummarizer | None = None,
        overwrite: bool = True
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
        self.vector_store = vector_store or VectorStore.from_chroma(config)
        self.doc_store = doc_store or DocumentStore.from_dict()
        self.table_summerizer= table_summerizer or TableSummarizer()
        self.markdown_chunker = chunk_markdown
        self.summerize_and_chunk_table = summerize_and_chunk_table
        self.store_chunk = store_chunks
        self.ingestion_dir = self.config.get('DOCS_DIR', 'docs')
        self.processed_dir = self.config.get('PROCESSED_DOCS_DIR', 'data/processed')
        self.store_persistence_path = self.config.get('DOCUMENT_STORE_PERSIST_PATH')
        self.table_context_window = int(self.config.get('TABLE_CONTEXT_WINDOW', 3))
        self.summarize_tables_enabled: bool = bool(self.config.get('SUMMARIZE_TABLES_ENABLED',True))
        logger.info(
            "IngestionPipeline initialized", 
            extra={
                "ingestion_dir": self.ingestion_dir,
                "processed_dir": self.processed_dir,
                "summarize_tables": self.summarize_tables_enabled,
            }
        )
        if overwrite:
            self.vector_store.reset_collection()
            logger.info("Vector DB collection reset")

    def run(self):
        """
        Executes the ingestion loop over all files in the source directory.

        For each file, it:
        1. Converts PDF to Markdown with enriched table context.
        2. Chunks the Markdown into hierarchical structures.
        3. Persists results into the vector and document stores.
        """
        logger.info("Starting ingestion run")
        for file_path in self._iter_files():
            logger.info("Processing file: %s", file_path)
            with Stopwatch('Markdown conversion'):
                markdown, tables = convert_pdf_to_markdown(
                    input_path=file_path,
                    table_context_window=self.table_context_window
                )
            logger.info(
                "Conversion complete",
                extra={"markdown_chars": len(markdown), "table_count": len(tables)}
            )

            table_chunks = self.summerize_and_chunk_table(
                tables,
                model=self.table_summerizer,
                summerize_enabled=self.summarize_tables_enabled,
            )
            logger.info("Table summarization complete", extra={"table_chunks": len(table_chunks)})

            chunks = self.markdown_chunker(
                markdown=markdown,
                source_path=file_path,
                config=self.config
            )
            logger.info("Chunked markdown into parent chunks", extra={"parent_chunks": len(chunks)})

            md_file_name = Path(self.processed_dir) / f"{Path(file_path).stem}.md"
            md_file_name.parent.mkdir(parents=True, exist_ok=True)
            md_file_name.write_text(markdown, encoding='utf-8')
            logger.info("Markdown snapshot stored", extra={"path": str(md_file_name)})

            total_chunks_before_extend = len(chunks)
            chunks.extend(table_chunks)
            logger.info(
                "Extended chunks with table entries",
                extra={
                    "doc_chunks": total_chunks_before_extend,
                    "table_chunks": len(table_chunks),
                    "total_chunks": len(chunks)
                }
            )

            self.store_chunk(
                chunks=chunks,
                vector_store=self.vector_store,
                doc_store=self.doc_store,
                persist_path=self.store_persistence_path
            )
            logger.info("Persisted chunk batch", extra={"chunk_count": len(chunks)})
        logger.info("Ingestion run completed")

    def _iter_files(self):
        """
        Yields file paths from the ingestion directory, skipping ignored items.

        Yields:
            str: Absolute path to the file.
        """
        for root, _, files in os.walk(self.ingestion_dir):
            for file in files:
                if file in self._IGNORE_DIR:
                    logger.debug(f"Ignored: {file}")
                    continue
                yield os.path.join(root, file)

if __name__ == '__main__':
    cfg = load_config()
    p = IngestionPipeline(config=cfg, overwrite=True)
    p.run()
