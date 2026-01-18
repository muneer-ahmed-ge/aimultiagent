from __future__ import annotations

import json
from typing import TypedDict, Literal, Optional, Dict, Any

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
    """Get work_order_type and product_id from work_order_id."""
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
    product_id: str
    work_order_type: str

    cleanup_steps: str
    final_answer: str


# -----------------------------
# LLM Router Node
# -----------------------------

def llm_router_node(state: AgentState) -> AgentState:
    """
    Uses AzureChatOpenAI to classify the user question and extract IDs.
    Returns:
      intent: schedule_flow | insights_only | knowledge_only | unknown
      work_order_id (optional)
      product_id (optional)
    """
    model = AzureChatOpenAI(deployment_name="SMAX-AI-Dev-GPT4")

    router_prompt = f"""
You are a routing classifier for a ServiceMax agent workflow.

Decide which workflow the system should run based on the user query.

Valid intents:
- schedule_flow: user asks what's scheduled today (and may also ask details/cleanup)
- insights_only: user asks for details/insights about a specific work order
- knowledge_only: user asks for cleanup steps/docs for a product
- unknown: none of the above

Also extract:
- work_order_id if present (example: WO1, WO-123)
- product_id if present (example: PROD-77881)

Return ONLY valid JSON with keys:
intent, work_order_id, product_id

User query:
{state["user_question"]}
""".strip()

    resp = model.invoke(router_prompt)

    # Parse JSON safely
    try:
        data = json.loads(resp.content)
    except Exception:
        data = {"intent": "unknown", "work_order_id": None, "product_id": None}

    intent = data.get("intent", "unknown")
    work_order_id = data.get("work_order_id")
    product_id = data.get("product_id")

    update: AgentState = {"intent": intent}

    # only set these if extracted
    if work_order_id:
        update["work_order_id"] = work_order_id
    if product_id:
        update["product_id"] = product_id

    return update


# -----------------------------
# Workflow Nodes
# -----------------------------

def scheduling_node(state: AgentState) -> AgentState:
    out = scheduling_service.invoke({})
    work_order_id = out.split("=")[1].strip()
    return {"work_order_id": work_order_id}


def service_insights_node(state: AgentState) -> AgentState:
    work_order_id = state.get("work_order_id")
    if not work_order_id:
        # If router failed to extract and we didn’t run scheduling
        return {"work_order_type": "Unknown", "product_id": ""}

    out = service_insights_service.invoke({"work_order_id": work_order_id})
    left = out.split(" for ")[0]
    parts = left.split(",")

    work_order_type = parts[0].split("=")[1].strip()
    product_id = parts[1].split("=")[1].strip()

    return {"work_order_type": work_order_type, "product_id": product_id}


def knowledge_access_node(state: AgentState) -> AgentState:
    product_id = state.get("product_id")
    if not product_id:
        return {"cleanup_steps": "No product_id available to fetch cleanup steps."}

    out = knowledge_access_service.invoke({"product_id": product_id})
    return {"cleanup_steps": out}


def final_answer_node(state: AgentState) -> AgentState:
    model = AzureChatOpenAI(deployment_name="SMAX-AI-Dev-GPT4")

    prompt = f"""
You are a ServiceMax assistant.
ONLY use the provided data and do not add assumptions.

User question:
{state.get("user_question")}

Known data:
- work_order_id: {state.get("work_order_id")}
- work_order_type: {state.get("work_order_type")}
- product_id: {state.get("product_id")}
- cleanup_steps: {state.get("cleanup_steps")}

Answer concisely.
""".strip()

    resp = model.invoke(prompt)
    return {"final_answer": resp.content}


# -----------------------------
# Conditional Routing Function
# -----------------------------

def route_from_intent(state: AgentState) -> str:
    return state["intent"]


# -----------------------------
# Build Graph
# -----------------------------

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("router", llm_router_node)
    graph.add_node("scheduling", scheduling_node)
    graph.add_node("service_insights", service_insights_node)
    graph.add_node("knowledge_access", knowledge_access_node)
    graph.add_node("final_answer", final_answer_node)

    graph.set_entry_point("router")

    # ✅ LLM-based routing
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

    # schedule flow: scheduling -> insights -> knowledge -> final
    graph.add_edge("scheduling", "service_insights")
    graph.add_edge("service_insights", "knowledge_access")
    graph.add_edge("knowledge_access", "final_answer")

    # insights only: insights -> final
    graph.add_edge("service_insights", "final_answer")

    # knowledge only: knowledge -> final
    graph.add_edge("knowledge_access", "final_answer")

    graph.add_edge("final_answer", END)

    return graph.compile()


# -----------------------------
# Run
# -----------------------------

def main():
    app = build_graph()

    queries = [
        "Provide details of work order WO1"
        # "What work order is scheduled today and how to clean the machine?",
        # "How do I clean product PROD-77881?",
    ]

    for q in queries:
        result = app.invoke({"user_question": q})
        print("\n==============================")
        print("USER:", q)
        print("------------------------------")
        print(result["final_answer"])


if __name__ == "__main__":
    main()
