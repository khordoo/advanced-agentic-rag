# OpenAI-Compatible API

The API exposes OpenAI-style routes under `/v1`.

## Chat Completions

Endpoint: `POST /v1/chat/completions`

The response always uses the OpenAI envelope (`id`, `object`, `created`, `model`, `choices`).
The assistant payload is always a string in `choices[0].message.content`.

### Response Format Header

Use the custom header `x-response-format` to control how the assistant content string is formatted.

Allowed values:

- `markdown` (default): human-friendly answer with a citations footer (recommended for OpenWebUI).
- `json`: JSON string containing structured fields such as `answer` and `citations`.

`x-response-format` is validated by the backend enum (`ResponseFormat`), so unsupported values are rejected.

### What Actually Changes

Only `choices[0].message.content` changes by format. The outer OpenAI response envelope stays the same.

```text
markdown -> "The answer...\n\n---\n**Sources:**\n* [1] ..."
json     -> "{\"answer\":\"The answer...\",\"citations\":[...]}"
```

### Example Requests

Markdown mode:

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "x-response-format: markdown" \
  -d '{
    "model": "advanced-rag-agent",
    "messages": [{"role": "user", "content": "Summarize the latest filing."}]
  }'
```

JSON-string mode:

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "x-response-format: json" \
  -d '{
    "model": "advanced-rag-agent",
    "messages": [{"role": "user", "content": "Summarize the latest filing."}]
  }'
```
