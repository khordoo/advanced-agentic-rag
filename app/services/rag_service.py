import json

from langgraph.graph.state import CompiledStateGraph

from app.schemas.openai import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
)


class RagService:
    def __init__(
        self,
        graph: CompiledStateGraph,
        system_instructions: str,
        drop_frontend_prompts: bool = True,
    ):
        self.graph = graph
        self.system_instructions = system_instructions
        self.drop_frontend_prompts = drop_frontend_prompts

    async def generate_chat_completion(
        self, chat_req: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Handles ALL data transformation, graph invocation, and response formatting."""

        langchain_msgs = chat_req.to_langchain_messages(
            drop_system_messages=self.drop_frontend_prompts
        )

        res = await self.graph.ainvoke(
            {"messages": langchain_msgs, "system_instructions": self.system_instructions}
        )

        agent_output_str = res["messages"][-1].content
        agent_output = json.loads(agent_output_str)
        # 1. Extract the actual text answer
        final_text = agent_output.get("answer", "")

        # 2. Format the citations cleanly at the bottom
        citations = agent_output.get("citations", [])
        if citations:
            final_text += "\n\n---\n**Sources:**\n"
            for idx, cite in enumerate(citations):
                # TODO: asset_name is empty for soem asset investiage. temporarily we use asset key in its place
                doc_name = cite.get("asset_key") or cite.get("asset_key", "Unknown Document")
                page = cite.get("page_number", "?")
                source_url = cite.get("source_url", "")
                if source_url:
                    # If it's a local Mac path, we prepend 'file://' so the browser recognizes it
                    href = f"file://{source_url}" if source_url.startswith("/") else source_url
                    final_text += f"* [{idx + 1}] [{doc_name} - Page {page}]({href})\n"
                else:
                    # Fallback if no URL exists
                    final_text += f"* [{idx + 1}] {doc_name} - Page {page}\n"

        return ChatCompletionResponse(
            model=chat_req.model,
            choices=[ChatChoice(message=ChatMessage(role="assistant", content=final_text))],
        )
