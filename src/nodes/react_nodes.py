"""LangGraph nodes for RAG workflow with ReAct agent inside generate_answer."""

from typing import List, Optional
from src.state.rag_state import RAGState

from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

# Wikipedia tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


class RAGNodes:
    """Node definitions for a 2-step RAG workflow with a ReAct agent."""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self._agent = None  # lazy init

    # ------------------------------------------------------------------
    # 1. RETRIEVE DOCUMENTS
    # ------------------------------------------------------------------
    def retrieve_docs(self, state: RAGState) -> RAGState:
        """Retrieve docs using the vectorstore retriever."""
        docs = self.retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=docs
        )

    # ------------------------------------------------------------------
    # 2. BUILD TOOLS (retriever + Wikipedia)
    # ------------------------------------------------------------------
    def _build_tools(self) -> List[Tool]:
        """Create tool wrappers for retriever + Wikipedia."""

        # Vectorstore retriever tool (wrapped to avoid type hint issues)
        def retriever_tool_fn(query: str) -> str:
            docs = self.retriever.invoke(query)
            if not docs:
                return "No documents found."

            merged = []
            for i, d in enumerate(docs[:8], start=1):
                meta = d.metadata or {}
                title = meta.get("title") or meta.get("source") or f"doc_{i}"
                merged.append(f"[{i}] {title}\n{d.page_content}")

            return "\n\n".join(merged)

        retriever_tool = Tool(
            name="retriever",
            description="Search in user-provided corpus.",
            func=retriever_tool_fn,
        )

        # Wikipedia tool (wrapped to prevent annotation-inspection issues)
        wiki = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=3, lang="en")
        )

        def wikipedia_tool_fn(query: str) -> str:
            return wiki.run(query)

        wikipedia_tool = Tool(
            name="wikipedia",
            description="Search Wikipedia for general world knowledge.",
            func=wikipedia_tool_fn,
        )

        return [retriever_tool, wikipedia_tool]

    # ------------------------------------------------------------------
    # 3. BUILD REACT AGENT
    # ------------------------------------------------------------------
    def _build_agent(self):
        """Construct the ReAct agent using LangChain's helper."""
        tools = self._build_tools()
        system_prompt = (
            "You are a helpful RAG agent.\n"
            "- Prefer the 'retriever' tool for document-based questions.\n"
            "- Use 'wikipedia' for external general knowledge.\n"
            "- Return only the final answer."
        )
        self._agent = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=system_prompt
        )

    # ------------------------------------------------------------------
    # 4. AGENT NODE
    # ------------------------------------------------------------------
    def generate_answer(self, state: RAGState) -> RAGState:
        """Generate answer using the ReAct agent."""
        if self._agent is None:
            self._build_agent()

        result = self._agent.invoke({
            "messages": [HumanMessage(content=state.question)]
        })

        # Extract last message content
        messages = result.get("messages", [])
        answer = messages[-1].content if messages else None

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer or "Could not generate answer."
        )
