from __future__ import annotations

from typing import TypedDict, Literal

from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END


# -----------------------------
# Tools
# -----------------------------

@tool
def scheduling_service() -> str:
    """Get scheduled work order today."""
    return "work_order_id=WO-100245"


@tool
def service_insights_service(work_order_id: str) -> str:
    """Get details + product_id from work_order_id."""
    return f"work_order_type=Critical,product_id=PROD-77881 for {work_order_id}"


@tool
def knowledge_access_service(product_id: str) -> str:
    """Get docs / cleanup steps from product_id."""
    return f"cleanup steps for {product_id}"


# -----------------------------
# State
# -----------------------------

Intent = Literal["schedule_flow", "insights_only", "knowledge_only", "unknown"]


class AgentState(TypedDict, total=False):
    user_question: str
    intent: Intent

    work_order_id: str
    work_order_type: str
    product_id: str

    cleanup_steps: str
    final_answer: str


# -----------------------------
# Nodes
# -----------------------------

def router_node(state: AgentState) -> AgentState:
    """
    In production, you can use an LLM classifier here.
    For now, we do simple keyword routing for clarity.
    """
    q = state["user_question"].lower()

    if "scheduled today" in q or "today" in q:
        return {"intent": "schedule_flow"}

    if "work order" in q and ("detail" in q or "details" in q or "insight" in q):
        return {"intent": "insights_only"}

    if "clean" in q or "cleanup" in q or "documentation" in q:
        return {"intent": "knowledge_only"}

    return {"intent": "unknown"}


def scheduling_node(state: AgentState) -> AgentState:
    out = scheduling_service.invoke({})
    work_order_id = out.split("=")[1].strip()
    return {"work_order_id": work_order_id}


def service_insights_node(state: AgentState) -> AgentState:
    # if user provided WO directly, state may already have it
    work_order_id = state.get("work_order_id", "WO-1")

    out = service_insights_service.invoke({"work_order_id": work_order_id})
    left = out.split(" for ")[0]
    parts = left.split(",")

    work_order_type = parts[0].split("=")[1].strip()
    product_id = parts[1].split("=")[1].strip()

    return {
        "work_order_type": work_order_type,
        "product_id": product_id,
        "work_order_id": work_order_id,
    }


def knowledge_access_node(state: AgentState) -> AgentState:
    # Requires product_id
    product_id = state.get("product_id")
    if not product_id:
        # Graph-safe fallback: no product_id yet
        return {"cleanup_steps": "No product_id available to fetch cleanup steps."}

    out = knowledge_access_service.invoke({"product_id": product_id})
    return {"cleanup_steps": out}


def final_answer_node(state: AgentState) -> AgentState:
    """
    LLM is optional here, but you can use AzureChatOpenAI
    to format a clean answer.
    """
    model = AzureChatOpenAI(deployment_name="SMAX-AI-Dev-GPT4")

    prompt = f"""
You are a ServiceMax assistant.
Only use the values below. Be concise.

user_question: {state.get("user_question")}

work_order_id: {state.get("work_order_id")}
work_order_type: {state.get("work_order_type")}
product_id: {state.get("product_id")}
cleanup_steps: {state.get("cleanup_steps")}

Return the best possible final answer based only on available fields.
""".strip()

    resp = model.invoke(prompt)
    return {"final_answer": resp.content}


# -----------------------------
# Conditional routing function
# -----------------------------

def route_from_intent(state: AgentState) -> str:
    return state["intent"]


# -----------------------------
# Build Graph
# -----------------------------

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("scheduling", scheduling_node)
    graph.add_node("service_insights", service_insights_node)
    graph.add_node("knowledge_access", knowledge_access_node)
    graph.add_node("final_answer", final_answer_node)

    graph.set_entry_point("router")

    # ✅ conditional edges based on router decision
    graph.add_conditional_edges(
        "router",
        route_from_intent,
        {
            "schedule_flow": "scheduling",
            "insights_only": "service_insights",
            "knowledge_only": "knowledge_access",
            "unknown": "final_answer",
        },
    )

    # schedule_flow path
    graph.add_edge("scheduling", "service_insights")
    graph.add_edge("service_insights", "knowledge_access")
    graph.add_edge("knowledge_access", "final_answer")

    # insights only path
    graph.add_edge("service_insights", "final_answer")

    # knowledge only path
    graph.add_edge("knowledge_access", "final_answer")

    graph.add_edge("final_answer", END)

    return graph.compile()


# -----------------------------
# Run
# -----------------------------

def main():
    app = build_graph()

    # Example 1: user asks only WO details
    q1 = "Provide the details of work order WO1"
    r1 = app.invoke({"user_question": q1, "work_order_id": "WO1"})
    print("\n--- Query 1 ---")
    print(r1["final_answer"])

    # Example 2: full workflow
    q2 = "What work order is scheduled today and how to clean the machine?"
    r2 = app.invoke({"user_question": q2})
    print("\n--- Query 2 ---")
    print(r2["final_answer"])


if __name__ == "__main__":
    main()
