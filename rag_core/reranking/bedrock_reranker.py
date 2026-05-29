from __future__ import annotations

import logging
import os

import boto3
from langchain_core.documents import Document

from rag_core.core.config import RerankerSettings
from rag_core.reranking.base import BaseReranker

logger = logging.getLogger(__name__)


class BedrockReranker(BaseReranker):
    """Reranker backed by the Amazon Bedrock Rerank API."""

    def __init__(self, settings: RerankerSettings | None = None) -> None:
        self._settings = settings or RerankerSettings()
        self._client = None

    def _load_client(self):
        if self._client is None:
            region_name = os.getenv("AWS_DEFAULT_REGION")
            client_kwargs = {"region_name": region_name} if region_name else {}
            self._client = boto3.client("bedrock-agent-runtime", **client_kwargs)
            logger.info(
                "Loaded Bedrock reranker client (region=%s, model=%s)",
                region_name or "default",
                self._settings.model_name,
            )
        return self._client

    def _resolve_model_arn(self) -> str:
        model_name = self._settings.model_name.strip()
        if model_name.startswith("arn:"):
            return model_name

        region_name = os.getenv("AWS_DEFAULT_REGION") or boto3.Session().region_name
        if not region_name:
            raise ValueError(
                "AWS_DEFAULT_REGION is required to build a Bedrock model ARN "
                f"from RERANK_MODEL_NAME={model_name!r}."
            )

        return f"arn:aws:bedrock:{region_name}::foundation-model/{model_name}"

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
    ) -> list[Document]:
        if not documents:
            return []

        client = self._load_client()
        actual_top_k = top_k if top_k is not None else self._settings.top_k

        response = client.rerank(
            queries=[
                {
                    "type": "TEXT",
                    "textQuery": {
                        "text": query,
                    },
                }
            ],
            sources=[
                {
                    "type": "INLINE",
                    "inlineDocumentSource": {
                        "type": "TEXT",
                        "textDocument": {"text": doc.page_content},
                    },
                }
                for doc in documents
            ],
            rerankingConfiguration={
                "type": "BEDROCK_RERANKING_MODEL",
                "bedrockRerankingConfiguration": {
                    "modelConfiguration": {
                        "modelArn": self._resolve_model_arn(),
                    },
                    "numberOfResults": actual_top_k,
                },
            },
        )

        for result in response.get("results", []):
            index = result["index"]
            documents[index].metadata["rerank_score"] = float(result["relevanceScore"])

        documents.sort(key=lambda d: d.metadata.get("rerank_score", 0), reverse=True)
        return documents[:actual_top_k]
