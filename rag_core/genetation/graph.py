from langchain_core import messages
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.system import SystemMessage
from langchain_core.messages.tool import ToolMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
import operator
import json
from langchain_core.documents import Document
from langgraph.prebuilt import ToolNode
from langsmith.schemas import Prompt
from langchain_core.tools import StructuredTool
from langchain_core.language_models import BaseChatModel


from enum import Enum
import os
import logging
from rag_core.agents import AgentFactory
from rag_core.genetation.tools import AgentToolFactory
from rag_core.genetation.nodes import NodeName, GraphNodes
from rag_core.genetation.state import RagState
from rag_core.prompts.loader import AgentPrompt
from rag_core.genetation.parsers import parse_agent_response
from rag_core.core.telemetry import get_observability_callbacks


logger = logging.getLogger(__name__)


class RagGraphFactory:
    @classmethod
    def build(
        cls,
        reasoning_llm: BaseChatModel | None = None,
        fast_llm: BaseChatModel | None = None,
        tools: list[StructuredTool] | None = None,
    ):
        logger.info("Initializing the graph")
        env = os.getenv("ENV", "dev")
        reasoning_llm = reasoning_llm or AgentFactory.get_default_model()
        fast_llm = fast_llm or AgentFactory.get_default_model()
        tools = tools or AgentToolFactory.create_tools()
        graph_nodes = GraphNodes(
            reasoning_llm=reasoning_llm,
            fast_llm=fast_llm,
            tools=tools,
        )
        graph: StateGraph = StateGraph(RagState)
        graph.add_node(
            NodeName.INTENT_ROUTER_AGENT, graph_nodes.execute_intent_router_agent
        )
        graph.add_node(
            NodeName.PRIMARY_RAG_AGENT, graph_nodes.execute_primary_rag_agent
        )
        graph.add_node(
            NodeName.GENERAL_ASSISTANT_AGENT,
            graph_nodes.execute_general_assistant_agent,
        )
        graph.add_node(NodeName.TOOLS, graph_nodes.execute_tools)
        graph.add_conditional_edges(
            NodeName.INTENT_ROUTER_AGENT,
            path=graph_nodes.route_after_triage,
            path_map={
                NodeName.PRIMARY_RAG_AGENT: NodeName.PRIMARY_RAG_AGENT,
                NodeName.GENERAL_ASSISTANT_AGENT: NodeName.GENERAL_ASSISTANT_AGENT,
            },
        )
        graph.add_conditional_edges(
            NodeName.PRIMARY_RAG_AGENT,
            path=graph_nodes.tool_router,
            path_map={NodeName.TOOLS: NodeName.TOOLS, NodeName.FINISH: END},
        )
        graph.add_edge(NodeName.TOOLS, NodeName.PRIMARY_RAG_AGENT)
        graph.add_edge(NodeName.GENERAL_ASSISTANT_AGENT, END)
        graph.set_entry_point(NodeName.INTENT_ROUTER_AGENT)

        g = graph.compile()
        if env == "dev":
            logger.info("Graph ASCII Layout: \n%s", g.get_graph().draw_ascii())
        logger.info("Graph initialized successfully")
        return g


if __name__ == "__main__":
    # query = "what is the price of organic egg? and what is the opening hours of the store?"
    query1 = "List the three specific macroeconomic or programmatic factors that caused the estimated fair value of the Gillette indefinite-lived intangible asset to drop, resulting in the $1.3 billion impairment."
    # "ground_truth": "The impairment was driven by a higher discount rate, the weakening of several currencies relative to the U.S. dollar, and the impact of the limited market portfolio restructuring program."
    query = "Detail the complete timeline of the limited market portfolio restructuring in Enterprise Markets. Include the initial estimated after-tax charges announced in December 2023, the specific liquidated charge as of June 2024, and the expected future charge for Q1 2025."
    # "ground_truth": "In December 2023, P&G announced expected incremental restructuring charges of $1.0 to $1.5 billion after tax. As of June 30, 2024, the company recorded a non-cash charge of $216 million after tax related to liquidating operations in Nigeria. For the first quarter of fiscal 2025, they expect to record a non-cash charge of approximately $750 million for accumulated currency translation losses following the divestiture of the Argentina business."
    query = "How are you?"
    # langfuse_handler = get_observability_callbacks()
    agent_prompt = AgentPrompt.from_yaml("rag_qa")
    llm = AgentFactory.create(provider="aws", model_name="amazon.nova-micro-v1:0")
    app = RagGraphFactory.build(llm=llm)

    # query = query1 + "Also," + query
    messages = []
    print("\n" + " @@@@" * 20)
    print("Interactive RAG Session. Type 'exit' to quit.")
    langfuse_callbacks = get_observability_callbacks()
    while True:
        try:
            query = input("\nUser > ")
        except (EOFError, KeyboardInterrupt):
            break

        if query.lower() in ["exit", "quit", "q"]:
            break
        if not query.strip():
            continue

        if not messages:
            # Critial: Reseting the Source should to empty list in the first invocation
            prompt = agent_prompt.chat_template.invoke(
                {"question": query, "source_docs": []}
            )
            print("=============Full PROMPT============\n")
            print(prompt)
            print("--" * 20)
            messages = prompt.to_messages()
            # print("==Prompt: ", messages)
        else:
            messages.append(HumanMessage(content=query))

        print("invoking the graph")
        res = app.invoke(
            {"messages": messages},
            config={
                "callbacks": langfuse_callbacks,
                "metadata": {"env": "local", "app": "dummy-langgraph-test"},
            },
        )

        messages = res["messages"]
        response_content = messages[-1].content
        print("=======" * 20)
        print("-----Final-----------")
        try:
            result = parse_agent_response(response_content)
            print(result.model_dump_json(indent=2))
            messages.append(AIMessage(content=result.model_dump_json()))
        except Exception as e:
            print("--" * 20)
            print("Error: ", e)
            print("Raw Response: ", response_content)
            print("--" * 20)
