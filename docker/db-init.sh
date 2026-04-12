#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username="$POSTGRES_USER" --dbname="$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE "$PERFECT_DB";
    CREATE DATABASE "$RAG_DB";
    CREATE DATABASE "$LANGFUSE_DB";
    -- Ownership is assigned automatically to $POSTGRES_USER
EOSQL
