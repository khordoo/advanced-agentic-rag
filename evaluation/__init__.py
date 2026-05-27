"""Evaluation module for the Advanced RAG pipeline.

This package is feature-flagged via ``EVAL_ENABLED`` (default ``false``).
All heavy dependencies (ragas, phoenix, phoenix.evals) are imported
lazily inside function bodies — ``import evaluation`` is always safe.
"""
