import logging
import os

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import StructuredTool
from langgraph.graph import END, StateGraph

from rag_core.agents import AgentFactory
from rag_core.core.config import GenerationSettings, get_generation_settings
from rag_core.genetation.nodes import GraphNodes, NodeName
from rag_core.genetation.state import RagState
from rag_core.genetation.tools import AgentToolFactory

logger = logging.getLogger(__name__)


class RagGraphFactory:
    @classmethod
    def build_default(cls):
        """
        Reads the config to figure out which models are assigned to which tiers,
        then asks the pure AgentFactory to build them.
        """
        logger.info("Initializing default graph dependencies from config...")
        settings = get_generation_settings()

        # We explicitly build the three different tiers here
        reasoning_model = AgentFactory.create(
            provider=settings.model_provider, model_name=settings.reasoning_model
        )
        verifier_model = AgentFactory.create(
            provider=settings.model_provider, model_name=settings.verifier_model
        )
        fast_model = AgentFactory.create(
            provider=settings.model_provider, model_name=settings.fast_model
        )

        return cls.build(
            reasoning_llm=reasoning_model,
            verifier_llm=verifier_model,
            fast_llm=fast_model,
            tools=AgentToolFactory.create_tools(),
            settings=settings,
        )

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
        settings = settings or get_generation_settings()
        graph_nodes = GraphNodes(
            reasoning_llm=reasoning_llm,
            verifier_llm=verifier_llm,
            fast_llm=fast_llm,
            tools=tools,
            settings=settings,
        )
        graph: StateGraph = StateGraph(RagState)
        graph.add_node(NodeName.INTENT_ROUTER_AGENT, graph_nodes.execute_intent_router_agent)
        graph.add_node(NodeName.PRIMARY_RAG_AGENT, graph_nodes.execute_primary_rag_agent)
        graph.add_node(
            NodeName.GENERAL_ASSISTANT_AGENT,
            graph_nodes.execute_general_assistant_agent,
        )
        graph.add_node(NodeName.SELF_CORRECTING_AGENT, graph_nodes.execute_self_correcting_agent)
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
        graph.set_conditional_entry_point(
            graph_nodes.route_initial_node,
            {
                NodeName.INTENT_ROUTER_AGENT: NodeName.INTENT_ROUTER_AGENT,
                NodeName.PRIMARY_RAG_AGENT: NodeName.PRIMARY_RAG_AGENT,
            },
        )
        logger.info("Conditional initial router configured.")

        g = graph.compile()
        if env == "dev":
            logger.info("Graph ASCII Layout: \n%s", g.get_graph().draw_ascii())
        logger.info("Graph initialized successfully")
        logger.info(
            f"Providers: {settings.model_provider}\n"
            f"Reasoning Model: {settings.reasoning_model}\n"
            f"Verifier Model: {settings.verifier_model}\n"
            f"Fast Model: {settings.fast_model}"
        )
        return g
