import logging
import os
from sqlite3.dbapi2 import Time
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from rag_core.converter import convert_pdf_to_markdown
from rag_core.chunker import chunk_markdown,store_chunks
from rag_core.store import VectorStore,DocumentStore
from rag_core.config import load_config
from rag_core.utils import Stopwatch
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IngestionPipeline:
    """Orchestrates the ingestion"""
    _IGNORE_DIR={'.DS_Store'}
    def __init__(self,config:dict[str,str],
        vector_store:VectorStore | None=None,
        doc_store: DocumentStore | None=None,overwrite:bool=True):
        self.config = config
        self.vector_store=vector_store or VectorStore.from_chroma(config)
        self.doc_store=doc_store or DocumentStore.from_dict()
        self.markdown_chunker=chunk_markdown
        self.store_chunk = store_chunks
        self.ingestion_dir = self.config.get('DOCS_DIR','docs')
        self.processed_dir = self.config.get('PROCCESED_DOCS_DIR', 'data/processed')
        self.store_persitence_path= self.config.get('DOCUMENT_STORE_PERSIST_PATH')
        if overwrite:
           self.vector_store.reset_collection()
           logger.info("Vector DB collection resetted")

    def run(self):

        for file_path in self._iter_files():
            logger.info(f'Processing files to Markdwn:{file_path}')
            with Stopwatch('Markdown conversion'):
                markdown, tables = convert_pdf_to_markdown(file_path)
                logger.debug(f"Conversion complete: {len(markdown)} chars and {len(tables)} Tables")
                chunks = self.markdown_chunker(
                    markdown=markdown,
                    source_path=file_path,
                    config=self.config
                )
                print('chunks:',chunks)
                logger.info(f"Chunked into {len(chunks)} parents.")
                self.store_chunk(
                    chunks=chunks,
                    vector_store=self.vector_store,
                    doc_store=self.doc_store,
                    persist_path=self.store_persitence_path
                )
                logger.info("Stored %d parent chunks in document store", len(chunks))


    def _iter_files(self):
        for root,_,files in os.walk(self.ingestion_dir):
            for file in files:
                if file in self._IGNORE_DIR:
                    print('ignored')
                    continue
                yield os.path.join(root,file)

if __name__=='__main__':
    cfg = load_config()
    # emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # vector_store = InMemoryVectorStore(emb)
    # doc_store: dict = {}

    p =IngestionPipeline(config=cfg,overwrite=True)
    p.run()
