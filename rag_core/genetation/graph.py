import os
import logging

from langgraph.graph import StateGraph, END
from langchain_core.tools import StructuredTool
from langchain_core.language_models import BaseChatModel

from rag_core.agents import AgentFactory
from rag_core.core.config import GenerationSettings
from rag_core.genetation.tools import AgentToolFactory
from rag_core.genetation.nodes import NodeName, GraphNodes
from rag_core.genetation.state import RagState


logger = logging.getLogger(__name__)


class RagGraphFactory:
    @classmethod
    def build(
        cls,
        reasoning_llm: BaseChatModel | None = None,
        verifier_llm: BaseChatModel | None = None,
        fast_llm: BaseChatModel | None = None,
        tools: list[StructuredTool] | None = None,
        settings: GenerationSettings | None = None,
    ):
        logger.info("Initializing the graph")
        env = os.getenv("ENV", "dev")
        reasoning_llm = reasoning_llm or AgentFactory.get_default_model()
        verifier_llm = verifier_llm or AgentFactory.get_default_model()
        fast_llm = fast_llm or AgentFactory.get_default_model()
        tools = tools or AgentToolFactory.create_tools()
        graph_nodes = GraphNodes(
            reasoning_llm=reasoning_llm,
            verifier_llm=verifier_llm,
            fast_llm=fast_llm,
            tools=tools,
            settings=settings,
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
        graph.add_node(
            NodeName.SELF_CORRECTING_AGENT, graph_nodes.execute_self_correcting_agent
        )
        graph.add_node(NodeName.POST_PROCESSING, graph_nodes.execute_post_processing)
        graph.add_node(NodeName.FALLBACK_HANDLER, graph_nodes.execute_fallback_handler)
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
            path_map={
                NodeName.TOOLS: NodeName.TOOLS,
                NodeName.FINISH: NodeName.SELF_CORRECTING_AGENT,
                NodeName.FALLBACK_HANDLER: NodeName.FALLBACK_HANDLER,
            },
        )
        graph.add_conditional_edges(
            NodeName.SELF_CORRECTING_AGENT,
            path=graph_nodes.verifier_router,
            path_map={
                NodeName.PRIMARY_RAG_AGENT: NodeName.PRIMARY_RAG_AGENT,
                NodeName.FALLBACK_HANDLER: NodeName.FALLBACK_HANDLER,
                NodeName.FINISH: NodeName.POST_PROCESSING,
            },
        )
        graph.add_edge(NodeName.TOOLS, NodeName.PRIMARY_RAG_AGENT)
        graph.add_edge(NodeName.GENERAL_ASSISTANT_AGENT, END)
        graph.add_edge(NodeName.FALLBACK_HANDLER, END)
        graph.add_edge(NodeName.POST_PROCESSING, END)
        graph.set_entry_point(NodeName.INTENT_ROUTER_AGENT)

        g = graph.compile()
        if env == "dev":
            logger.info("Graph ASCII Layout: \n%s", g.get_graph().draw_ascii())
        logger.info("Graph initialized successfully")
        return g
