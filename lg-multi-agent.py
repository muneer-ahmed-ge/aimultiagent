from __future__ import annotations

from typing import TypedDict, Optional

from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI

from langgraph.graph import StateGraph, END


# -----------------------------
# Tools (same as your LangChain)
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
# Graph State
# -----------------------------

class AgentState(TypedDict, total=False):
    user_question: str

    work_order_id: str
    work_order_type: str
    product_id: str

    cleanup_steps: str
    final_answer: str


# -----------------------------
# Nodes
# -----------------------------

def scheduling_node(state: AgentState) -> AgentState:
    out = scheduling_service.invoke({})
    work_order_id = out.split("=")[1].strip()
    return {"work_order_id": work_order_id}


def service_insights_node(state: AgentState) -> AgentState:
    work_order_id = state["work_order_id"]

    out = service_insights_service.invoke({"work_order_id": work_order_id})
    # "work_order_type=Critical,product_id=PROD-77881 for WO-100245"
    left = out.split(" for ")[0]
    parts = left.split(",")

    work_order_type = parts[0].split("=")[1].strip()
    product_id = parts[1].split("=")[1].strip()

    return {"work_order_type": work_order_type, "product_id": product_id}


def knowledge_access_node(state: AgentState) -> AgentState:
    product_id = state["product_id"]
    out = knowledge_access_service.invoke({"product_id": product_id})
    return {"cleanup_steps": out}


def llm_final_answer_node(state: AgentState) -> AgentState:
    """
    Use AzureChatOpenAI to generate a grounded final response
    using only the tool outputs in state.
    """
    model = AzureChatOpenAI(deployment_name="SMAX-AI-Dev-GPT4")

    prompt = f"""
You are a ServiceMax field service assistant.
Only use the provided data below. Do not add assumptions.

User question:
{state["user_question"]}

Tool outputs:
- work_order_id: {state["work_order_id"]}
- work_order_type: {state["work_order_type"]}
- product_id: {state["product_id"]}
- cleanup_steps: {state["cleanup_steps"]}

Return a concise final answer including:
1) Work order id
2) Work order type
3) Cleanup steps
""".strip()

    response = model.invoke(prompt)
    return {"final_answer": response.content}


# -----------------------------
# Build Graph
# -----------------------------

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("scheduling", scheduling_node)
    graph.add_node("service_insights", service_insights_node)
    graph.add_node("knowledge_access", knowledge_access_node)
    graph.add_node("final_answer_llm", llm_final_answer_node)

    graph.set_entry_point("scheduling")
    graph.add_edge("scheduling", "service_insights")
    graph.add_edge("service_insights", "knowledge_access")
    graph.add_edge("knowledge_access", "final_answer_llm")
    graph.add_edge("final_answer_llm", END)

    return graph.compile()


# -----------------------------
# Run
# -----------------------------

def main():
    user_question = "What work order is scheduled today give me its id and type and how to clean the machine?"

    app = build_graph()

    result = app.invoke({"user_question": user_question})
    print("\n--- FINAL OUTPUT ---\n")
    print(result["final_answer"])


if __name__ == "__main__":
    main()
