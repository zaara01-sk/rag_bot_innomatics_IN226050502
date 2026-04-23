"""
Graph Workflow Module
LangGraph-based flow: input -> retrieve -> generate -> route -> output / escalate
Includes HITL (Human-in-the-Loop) escalation logic.
"""

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# State object that flows through all nodes.
# Every node reads from this dict and writes back to it.
class SupportState(TypedDict):
    query: str
    retrieved_chunks: list[str]
    answer: str
    confidence: str      # "high" | "low"
    needs_human: bool
    human_response: Optional[str]
    final_output: str


LOW_CONFIDENCE_KEYWORDS = [
    "not sure", "cannot", "don't know", "no information",
    "unclear", "unable to find", "outside my knowledge"
]

COMPLEX_KEYWORDS = [
    "legal", "refund policy", "lawsuit", "billing dispute",
    "escalate", "manager", "complaint", "fraud"
]


def retrieve_node(state: SupportState, retriever) -> SupportState:
    query = state["query"]
    docs = retriever.invoke(query)
    chunks = [doc.page_content for doc in docs]
    print(f"[retrieve_node] Got {len(chunks)} chunks for query: '{query[:60]}...'")
    return {**state, "retrieved_chunks": chunks}


def generate_node(state: SupportState, llm) -> SupportState:
    context = "\n\n".join(state["retrieved_chunks"])
    query = state["query"]

    # Prompt I wrote after a few iterations.
    # The explicit "say you don't know" instruction is important for RAG honesty.
    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful customer support assistant. Use ONLY the context below to answer the customer's question.
        If the answer is not in the context, say clearly that you don't have that information.
        Do not make things up.
        Context:
        {context}
        Customer Question: {query}
        Answer:
        """)

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "query": query})

    # Simple heuristic to gauge confidence - not perfect but works for this scope
    answer_lower = answer.lower()
    is_low_confidence = any(kw in answer_lower for kw in LOW_CONFIDENCE_KEYWORDS)
    confidence = "low" if is_low_confidence else "high"

    print(f"[generate_node] Confidence: {confidence}")
    return {**state, "answer": answer, "confidence": confidence}


def route_node(state: SupportState) -> SupportState:
    query_lower = state["query"].lower()

    # escalate if: low confidence, complex/sensitive query, or no context found
    is_complex = any(kw in query_lower for kw in COMPLEX_KEYWORDS)
    is_low_conf = state["confidence"] == "low"
    no_context = len(state["retrieved_chunks"]) == 0

    needs_human = is_complex or is_low_conf or no_context

    if needs_human:
        reason = []
        if is_complex: reason.append("complex/sensitive query")
        if is_low_conf: reason.append("low confidence answer")
        if no_context: reason.append("no relevant documents found")
        print(f"[route_node] Escalating to human. Reasons: {', '.join(reason)}")
    else:
        print("[route_node] Routing to direct answer output")

    return {**state, "needs_human": needs_human}


def hitl_node(state: SupportState) -> SupportState:
    # simulating HITL with a CLI prompt — in production this would be a ticketing API
    print("\n" + "="*60)
    print("⚠️ ESCALATION: Human Agent Required")
    print("="*60)
    print(f"Customer Query: {state['query']}")
    print(f"AI Draft Answer: {state['answer']}")
    print("-"*60)

    # In production: send to ticketing system (Zendesk, Freshdesk etc.)
    # and await callback. For now, simulate agent response.
    try:
        human_input = input("Human Agent - Enter your response (or press Enter to accept AI draft): ").strip()
    except EOFError:
        # Non-interactive environment (e.g. tests)
        human_input = ""

    final = human_input if human_input else state["answer"]
    final_output = f"[Reviewed by Human Agent]\n{final}"
    return {**state, "human_response": human_input, "final_output": final_output}


def output_node(state: SupportState) -> SupportState:
    final_output = f"[AI Assistant]\n{state['answer']}"
    return {**state, "final_output": final_output}


def should_escalate(state: SupportState) -> str:
    return "hitl" if state["needs_human"] else "output"


def build_graph(retriever, llm):
    graph = StateGraph(SupportState)

    # Wrap nodes to inject dependencies (retriever, llm)
    graph.add_node("retrieve", lambda s: retrieve_node(s, retriever))
    graph.add_node("generate", lambda s: generate_node(s, llm))
    graph.add_node("route", route_node)
    graph.add_node("hitl", hitl_node)
    graph.add_node("output", output_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "route")

    # Conditional routing: escalate to human or go to output
    graph.add_conditional_edges(
        "route",
        should_escalate,
        {"hitl": "hitl", "output": "output"}
    )

    graph.add_edge("hitl", END)
    graph.add_edge("output", END)

    return graph.compile()
