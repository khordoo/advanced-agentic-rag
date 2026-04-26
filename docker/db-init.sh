#!/bin/sh
set -e

die() {
    echo "ERROR: $1" >&2
    exit 1
}

require_non_empty() {
    var_name="$1"
    value="$2"
    [ -n "$value" ] || die "Required env var '$var_name' is empty or missing. Set it in your .env file."
}

create_db() {
    db_name="$1"
    psql -v ON_ERROR_STOP=1 --username="$POSTGRES_USER" --dbname="$POSTGRES_DB" \
        -c "CREATE DATABASE \"$db_name\";"
    echo "[INFO] databse created (db=$db_name)"
}
create_extension(){
    db_name="$1"
    extension_name="$2"
    psql -v ON_ERROR_STOP=1 --username="$POSTGRES_USER" --dbname="$db_name" \
        -c "CREATE EXTENSION IF NOT EXISTS \"$extension_name\";"
    echo "[INFO] extension initialized (db=$db_name, ext=$extension_name)"
}

RAG_DB_NAME="${RAG_DB:-${POSTGRES_RAG_DB:-}}"

require_non_empty "POSTGRES_USER" "${POSTGRES_USER:-}"
require_non_empty "POSTGRES_DB" "${POSTGRES_DB:-}"
require_non_empty "PERFECT_DB" "${PERFECT_DB:-}"
require_non_empty "RAG_DB (or POSTGRES_RAG_DB)" "$RAG_DB_NAME"
require_non_empty "LANGFUSE_DB" "${LANGFUSE_DB:-}"

create_db "${PERFECT_DB}"
create_db "$RAG_DB_NAME"
create_db "${LANGFUSE_DB}"

create_extension "$RAG_DB_NAME" "vector"
