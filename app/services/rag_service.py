import json
import logging
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from app.core.exceptions import UpstreamModelError
from app.schemas.openai import ChatCompletionRequest
from app.services.formatters import ChatPresenter

logger = logging.getLogger(__name__)


class GenerationResult(BaseModel):
    """
    The standardized output from the AI reasoning layer.
    Extensible for future observability and routing metrics.
    """

    requested_model_id: str
    resolved_model_id: str
    agent_output: dict[str, Any]


class RagService:
    def __init__(
        self,
        graph: CompiledStateGraph,
        system_instructions: str,
        model_id: str,
        drop_frontend_prompts: bool = True,
    ):
        self.graph = graph
        self.system_instructions = system_instructions
        self.model_id = model_id
        self.drop_frontend_prompts = drop_frontend_prompts

    @staticmethod
    def _extract_chunk_text(chunk: Any) -> str:
        content = getattr(chunk, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts)
        return ""

    async def generate_chat_completion(self, chat_req: ChatCompletionRequest) -> GenerationResult:
        """Runs graph inference and returns canonical output + resolved model metadata."""

        langchain_msgs = chat_req.to_langchain_messages(
            drop_system_messages=self.drop_frontend_prompts
        )

        try:
            res = await self.graph.ainvoke(
                {"messages": langchain_msgs, "system_instructions": self.system_instructions}
            )
            messages = res.get("messages")
            if not isinstance(messages, list) or not messages:
                raise UpstreamModelError("Upstream model output is missing messages.")

            final_message = messages[-1]
            content = getattr(final_message, "content", None)
            if not isinstance(content, str) or not content.strip():
                raise UpstreamModelError("Upstream model output content is empty or invalid.")

            parsed_output = json.loads(content)
            if not isinstance(parsed_output, dict):
                raise UpstreamModelError("Upstream model output must be a JSON object.")
        except UpstreamModelError:
            raise
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as exc:
            raise UpstreamModelError("Upstream model returned malformed JSON output.") from exc
        except Exception as exc:
            raise UpstreamModelError("Upstream model invocation failed.") from exc

        return GenerationResult(
            requested_model_id=chat_req.model,
            resolved_model_id=self.model_id,
            agent_output=parsed_output,
        )

    async def stream_chat_completion(
        self,
        chat_req: ChatCompletionRequest,
        response_format: str,
    ):
        """
        Streams OpenAI-compatible content deltas from LangGraph events.
        """
        langchain_msgs = chat_req.to_langchain_messages(
            drop_system_messages=self.drop_frontend_prompts
        )
        stream = self.graph.astream_events(
            {"messages": langchain_msgs, "system_instructions": self.system_instructions},
            version="v2",
        )

        parser = _IncrementalAnswerJSONParser()
        full_json_buffer: list[str] = []
        emitted_any_token = False

        try:
            async for event in stream:
                if event.get("event") != "on_chat_model_stream":
                    continue
                data = event.get("data") or {}
                chunk = data.get("chunk")
                if chunk is None:
                    continue

                delta = self._extract_chunk_text(chunk)
                if not delta:
                    continue
                full_json_buffer.append(delta)

                if response_format == "json":
                    emitted_any_token = True
                    yield delta
                    continue

                answer_delta = parser.feed(delta)
                if answer_delta:
                    emitted_any_token = True
                    yield answer_delta
        except UpstreamModelError:
            raise
        except Exception as exc:
            if emitted_any_token:
                logger.exception("streaming_failed_midstream", extra={"error": type(exc).__name__})
                yield "\n\n*[System Error: Upstream model connection lost.]*"
                return
            raise UpstreamModelError("Upstream model streaming invocation failed.") from exc

        raw_json = "".join(full_json_buffer).strip()
        if not raw_json:
            if emitted_any_token:
                return
            raise UpstreamModelError("Upstream model stream produced no content.")

        try:
            parsed = json.loads(raw_json)
            if not isinstance(parsed, dict):
                raise UpstreamModelError("Upstream streamed output must be a JSON object.")
        except UpstreamModelError:
            raise
        except Exception as exc:
            if emitted_any_token:
                logger.exception("streaming_parse_failed_midstream", extra={"error": type(exc).__name__})
                yield "\n\n*[System Error: Upstream model response was malformed.]*"
                return
            raise UpstreamModelError("Upstream model streamed malformed JSON output.") from exc

        if response_format != "json":
            markdown_full = ChatPresenter.to_markdown(parsed)
            answer_only = parsed.get("answer", "")
            if not isinstance(answer_only, str):
                answer_only = str(answer_only)
            trailing = markdown_full[len(answer_only) :] if markdown_full.startswith(answer_only) else ""
            if trailing:
                yield trailing


class _IncrementalAnswerJSONParser:
    """
    Extracts only the answer string value from streamed JSON fragments.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._started = False
        self._in_answer = False
        self._escape = False

    def feed(self, fragment: str) -> str:
        self._buffer += fragment
        out: list[str] = []
        idx = 0

        while idx < len(self._buffer):
            if not self._started:
                marker_index = self._buffer.find('"answer"', idx)
                if marker_index == -1:
                    self._buffer = self._buffer[-16:]
                    return ""
                idx = marker_index + len('"answer"')

                while idx < len(self._buffer) and self._buffer[idx].isspace():
                    idx += 1
                if idx >= len(self._buffer):
                    self._buffer = self._buffer[marker_index:]
                    return ""
                if self._buffer[idx] != ":":
                    continue
                idx += 1
                while idx < len(self._buffer) and self._buffer[idx].isspace():
                    idx += 1
                if idx >= len(self._buffer):
                    self._buffer = self._buffer[marker_index:]
                    return ""
                if self._buffer[idx] != '"':
                    # answer is not a string; bail out silently
                    self._buffer = self._buffer[idx:]
                    return ""
                self._started = True
                self._in_answer = True
                self._escape = False
                idx += 1
                continue

            if self._in_answer:
                ch = self._buffer[idx]
                if self._escape:
                    out.append(ch)
                    self._escape = False
                    idx += 1
                    continue
                if ch == "\\":
                    self._escape = True
                    idx += 1
                    continue
                if ch == '"':
                    self._in_answer = False
                    self._buffer = self._buffer[idx + 1 :]
                    return "".join(out)
                out.append(ch)
                idx += 1
                continue

            break

        if self._in_answer:
            self._buffer = ""
        return "".join(out)
