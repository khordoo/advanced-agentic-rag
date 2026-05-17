import json
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from app.core.exceptions import UpstreamModelError
from app.schemas.openai import ChatCompletionRequest


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
