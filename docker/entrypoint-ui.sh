#!/bin/sh
set -e

if [ "${UI_APP:-fastapi}" = "streamlit" ]; then
    echo "Launching Streamlit Chunking Lab..."
    exec .venv/bin/streamlit run examples/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
fi

echo "Launching FastAPI Web Chat Server..."
exec .venv/bin/uvicorn rag_core.web.server:app --host 0.0.0.0 --port 8501
