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

        rag_response = res["messages"][-1].content

        return ChatCompletionResponse(
            model=chat_req.model,
            choices=[ChatChoice(message=ChatMessage(role="assistant", content=rag_response))],
        )
