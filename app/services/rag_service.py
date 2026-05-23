import json
import logging
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from app.core.exceptions import UpstreamModelError
from app.schemas.openai import ChatCompletionRequest
from app.services.formatters import ChatPresenter
from app.services.stream_adapter import adapt_graph_stream
from rag_core.core.readiness import ApplicationReadinessChecker

logger = logging.getLogger(__name__)


class _IncrementalAnswerJSONParser:
    """Extract incremental characters from a streaming JSON ``answer`` field."""

    def __init__(self) -> None:
        self._buffer = ""
        self._emitted = ""

    def feed(self, chunk: str) -> str:
        self._buffer += chunk
        marker = '"answer"'
        marker_index = self._buffer.find(marker)
        if marker_index == -1:
            return ""

        colon_index = self._buffer.find(":", marker_index + len(marker))
        if colon_index == -1:
            return ""

        value_start = self._buffer.find('"', colon_index + 1)
        if value_start == -1:
            return ""

        answer_chars: list[str] = []
        escaped = False
        for char in self._buffer[value_start + 1 :]:
            if escaped:
                if char == '"':
                    answer_chars.append('"')
                elif char == "\\":
                    answer_chars.append("\\")
                elif char == "n":
                    answer_chars.append("\n")
                elif char == "t":
                    answer_chars.append("\t")
                else:
                    answer_chars.append(char)
                escaped = False
                continue

            if char == "\\":
                escaped = True
                continue
            if char == '"':
                break
            answer_chars.append(char)

        answer = "".join(answer_chars)
        delta = answer[len(self._emitted) :]
        self._emitted = answer
        return delta


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
        readiness_checker: ApplicationReadinessChecker | None = None,
    ):
        self.graph = graph
        self.system_instructions = system_instructions
        self.model_id = model_id
        self.drop_frontend_prompts = drop_frontend_prompts
        self.readiness_checker = readiness_checker

    async def generate_chat_completion(self, chat_req: ChatCompletionRequest) -> GenerationResult:
        """Runs graph inference and returns canonical output + resolved model metadata."""
        self._ensure_ready()

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

    def create_event_stream(self, chat_req: ChatCompletionRequest):
        """Returns a raw LangGraph event stream for the adapter layer to consume.

        The returned async iterator yields ``astream_events(v2)`` dicts.
        The caller (``adapt_graph_stream``) is responsible for filtering
        events and formatting output.
        """
        self._ensure_ready()
        langchain_msgs = chat_req.to_langchain_messages(
            drop_system_messages=self.drop_frontend_prompts
        )
        return self.graph.astream_events(
            {"messages": langchain_msgs, "system_instructions": self.system_instructions},
            version="v2",
        )

    async def stream_chat_completion(
        self,
        chat_req: ChatCompletionRequest,
        response_format: str = "markdown",
    ):
        event_stream = self.create_event_stream(chat_req)
        parser = _IncrementalAnswerJSONParser()
        raw_json = ""
        saw_token_stream = False
        buffered_events: list[dict[str, Any]] = []

        async for event in event_stream:
            buffered_events.append(event)
            if event.get("event") != "on_chat_model_stream":
                continue
            saw_token_stream = True
            chunk = event.get("data", {}).get("chunk")
            content = getattr(chunk, "content", None)
            if not isinstance(content, str):
                continue
            raw_json += content
            delta = parser.feed(content)
            if delta:
                yield delta

        if not saw_token_stream:
            async def _replay_events():
                for event in buffered_events:
                    yield event

            async for chunk in adapt_graph_stream(
                _replay_events(),
                response_format=response_format,
            ):
                yield chunk
            return

        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError:
            yield "\n\n*[System Error: completion failed.]*"
            return

        if response_format == "json":
            yield ChatPresenter.to_json_string(parsed)
        else:
            yield "\n" + ChatPresenter.to_markdown(parsed)

    def _ensure_ready(self) -> None:
        if self.readiness_checker is not None:
            self.readiness_checker.ensure_ready()
