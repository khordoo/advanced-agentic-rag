import logging
import uuid
from typing import Any, Dict, List, Iterable
from qdrant_client import QdrantClient, models
from rag_core.indexing.base import BaseIndexer
from rag_core.db.connection_manager import QdrantConnectionManager
from rag_core.config import get_qdrant_settings, QdrantSettings
from rag_core.schemas import PARENT_INDEX_SCHEMA, CHILD_INDEX_SCHEMA, ParentChunk


logger = logging.getLogger(__name__)


class QdrantIndexer(BaseIndexer):

    def __init__(self, client: QdrantClient | None = None, settings: QdrantSettings | None = None):
        self.client = client or QdrantConnectionManager.get_client()
        self.settings = settings or get_qdrant_settings()
        namespace = self.settings.collection_namespace  # small chunks for vector

        self.chunk_collection_name = f"{namespace}_chunks"  # e.g., 'docs_chunks'
        self.parent_collection_name = f"{namespace}_parents"  # e.g., 'docs_parents'

        self._initialize_collections()


    def _initialize_collections(self ):
        self._create_collection(self.chunk_collection_name, CHILD_INDEX_SCHEMA,is_vector=True)
        self._create_collection(self.parent_collection_name, PARENT_INDEX_SCHEMA, is_vector=False)

    # The asterisk means everything after it MUST be passed by name
    def _create_collection(self, name: str, schema_registry: dict[str, Any],*, is_vector:bool = True):
        if not self.client.collection_exists(name):
            logger.info(f"Creating collection '{name}'...")

            if is_vector:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=self.client.get_fastembed_vector_params(),
                    sparse_vectors_config=self.client.get_fastembed_sparse_vector_params()
                )
            else:
                #document store, no vector search is needed
                self.client.create_collection(
                    collection_name=name,
                    vectors_config={},
                )

            #creating index
            for field_name,field_type in schema_registry.items():
                self.client.create_payload_index(
                    collection_name=name,
                    field_name=field_name,
                    field_schema=field_type
                )
            logger.info(f"Index applied for {name}.")

    def ingest_documents(self, document_blocks:list[ParentChunk]):
        """Pushes new documents into the vector database using modern upsert."""
        logger.info(f"Ingesting {len(document_blocks)} chunks into '{self.chunk_collection_name}'...")
        vector_name = self.client.get_vector_field_name()
        sparse_vector_name = self.client.get_sparse_vector_field_name()
        # TODO: pass the chunk object use child text for vector and save parent text in the metadata object or parent id? but we would need a different document store?  or a seperate index in qdrant that only save parents?
        parent_points=[]
        child_points=[]
        for block in document_blocks:
            parent_points.append(
                models.PointStruct(
                    id=block.id,
                    vector={},
                    payload={'text':block.enriched_text,**block.metadata.model_dump()}
                )
            )
            for child_doc in block.children:
                child_points.append(
                    models.PointStruct(
                        id=(str(uuid.uuid4())),
                        vector={
                            # Dense embedding (uses your BAAI model)
                            vector_name: models.Document(
                                text=child_doc.page_content,
                                model=self.settings.embed_model
                            ),
                            # Sparse embedding (uses your Splade model)
                            sparse_vector_name: models.Document(
                                text=child_doc.page_content,
                                model=self.settings.sparse_model
                            )
                        },
                        payload=child_doc.metadata
                    )
                )

        logger.info(f"Ingesting document chunks into the qdrant database...")
        self.client.upsert(
            collection_name=self.chunk_collection_name,
            points=child_points
        )
        logger.info(f"Ingesting document parents into the qdrant database...")
        self.client.upsert(
            collection_name=self.parent_collection_name,
            points=parent_points
        )
        logger.info(f"Ingestion completed: {len(parent_points)} parents, {len(child_points)} child chunks, into {self.chunk_collection_name} and {self.parent_collection_name} indexes.")

    def reset_collections(self):
        logger.info(f"Resetting collection indexes...")
        self._delete_collection(self.chunk_collection_name)
        self._delete_collection(self.parent_collection_name)
        self._initialize_collections()

    def _delete_collection(self, collection_name: str):
        if self.client.collection_exists(collection_name):
            self.client.delete_collection(collection_name=collection_name)
            logger.info(f"Deleted collection '{collection_name}'.")
        else:
            logger.info(f"Collection '{collection_name}' does not exist. Deletion skipped.")