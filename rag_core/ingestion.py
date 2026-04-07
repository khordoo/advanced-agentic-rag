import logging
import os
from rag_core.converter import convert_pdf_to_markdown
from rag_core.chunker import chunk_markdown, store_chunks
from rag_core.store import VectorStore, DocumentStore
from rag_core.config import load_config, setup_logging
from rag_core.utils import Stopwatch

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
        self.markdown_chunker = chunk_markdown
        self.store_chunk = store_chunks
        self.ingestion_dir = self.config.get('DOCS_DIR', 'docs')
        self.processed_dir = self.config.get('PROCESSED_DOCS_DIR', 'data/processed')
        self.store_persistence_path = self.config.get('DOCUMENT_STORE_PERSIST_PATH')
        self.table_context_window = int(self.config.get('TABLE_CONTEXT_WINDOW', 3))
        
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
        for file_path in self._iter_files():
            logger.info(f'Processing file: {file_path}')
            with Stopwatch('Markdown conversion'):
                markdown, tables = convert_pdf_to_markdown(
                    input_path=file_path,
                    table_context_window=self.table_context_window
                )
                logger.debug(f"Conversion complete: {len(markdown)} chars and {len(tables)} tables")
                
                chunks = self.markdown_chunker(
                    markdown=markdown,
                    source_path=file_path,
                    config=self.config
                )
                logger.info(f"Chunked into {len(chunks)} parents.")
                
                self.store_chunk(
                    chunks=chunks,
                    vector_store=self.vector_store,
                    doc_store=self.doc_store,
                    persist_path=self.store_persistence_path
                )
                logger.info("Stored %d parent chunks in document store", len(chunks))

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
