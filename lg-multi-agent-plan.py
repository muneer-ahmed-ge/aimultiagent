from __future__ import annotations

import json
from typing import TypedDict, List, Literal, Optional

from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END


# -----------------------------
# Tools (same as before)
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

Step = Literal[
    "scheduling",
    "service_insights",
    "knowledge_access",
    "final_answer",
]


class AgentState(TypedDict, total=False):
    user_question: str

    # produced by router
    plan: List[Step]

    # working memory
    work_order_id: Optional[str]
    work_order_type: Optional[str]
    product_id: Optional[str]
    cleanup_steps: Optional[str]

    final_answer: Optional[str]


# -----------------------------
# Router => produces PLAN
# -----------------------------

def llm_planner_node(state: AgentState) -> AgentState:
    model = AzureChatOpenAI(deployment_name="SMAX-AI-Dev-GPT4")

    prompt = f"""
You are a planner for a ServiceMax workflow agent.

Given the user query, return a plan as a JSON object with key "plan".

Allowed steps (in correct dependency order):
- "scheduling" (gets today's work_order_id)
- "service_insights" (needs work_order_id -> gets work_order_type and product_id)
- "knowledge_access" (needs product_id -> gets cleanup steps)
- "final_answer" (compose final response)

Rules:
- If the user asks about "scheduled today" you MUST include "scheduling".
- If the user asks for work order details/type, include "service_insights".
- If the user asks cleanup/docs, include "knowledge_access".
- Always include "final_answer" as the last step.
- If user provides a work_order_id directly, you may skip "scheduling".
- If user provides a product_id directly, you may skip earlier steps.

Return ONLY valid JSON like:
{{"plan": ["service_insights", "final_answer"]}}

User query:
{state["user_question"]}
""".strip()

    resp = model.invoke(prompt)

    try:
        data = json.loads(resp.content)
        plan = data.get("plan", [])
    except Exception:
        # safe fallback
        plan = ["final_answer"]

    # Ensure final_answer is always last
    if not plan or plan[-1] != "final_answer":
        plan.append("final_answer")

    return {"plan": plan}


# -----------------------------
# Step execution nodes
# -----------------------------

def scheduling_node(state: AgentState) -> AgentState:
    out = scheduling_service.invoke({})
    work_order_id = out.split("=")[1].strip()
    return {"work_order_id": work_order_id}


def service_insights_node(state: AgentState) -> AgentState:
    work_order_id = state.get("work_order_id")
    if not work_order_id:
        return {"work_order_type": "Unknown", "product_id": None}

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
ONLY use the provided values. Do not add assumptions.

User question:
{state.get("user_question")}

Known values:
work_order_id: {state.get("work_order_id")}
work_order_type: {state.get("work_order_type")}
product_id: {state.get("product_id")}
cleanup_steps: {state.get("cleanup_steps")}

Return a concise final response.
""".strip()

    resp = model.invoke(prompt)
    return {"final_answer": resp.content}


# -----------------------------
# Loop router: pick next step
# -----------------------------

def choose_next_step(state: AgentState) -> str:
    """
    Pops the next step from the plan and routes to that node.
    If plan is empty => END.
    """
    plan = state.get("plan", [])
    if not plan:
        return END

    next_step = plan[0]
    return next_step


def pop_plan_step(state: AgentState) -> AgentState:
    """
    Remove the first step after it runs.
    """
    plan = state.get("plan", [])
    if plan:
        plan = plan[1:]
    return {"plan": plan}


# -----------------------------
# Build Graph
# -----------------------------

def build_graph():
    g = StateGraph(AgentState)

    # Nodes
    g.add_node("planner", llm_planner_node)

    g.add_node("scheduling", scheduling_node)
    g.add_node("service_insights", service_insights_node)
    g.add_node("knowledge_access", knowledge_access_node)
    g.add_node("final_answer", final_answer_node)

    # helper node to pop plan steps after each execution
    g.add_node("pop_step", pop_plan_step)

    # Entry
    g.set_entry_point("planner")

    # After planning, start executing steps (loop)
    g.add_conditional_edges(
        "planner",
        choose_next_step,
        {
            "scheduling": "scheduling",
            "service_insights": "service_insights",
            "knowledge_access": "knowledge_access",
            "final_answer": "final_answer",
            END: END,
        },
    )

    # After each step, pop it then decide again
    for step in ["scheduling", "service_insights", "knowledge_access", "final_answer"]:
        g.add_edge(step, "pop_step")

    g.add_conditional_edges(
        "pop_step",
        choose_next_step,
        {
            "scheduling": "scheduling",
            "service_insights": "service_insights",
            "knowledge_access": "knowledge_access",
            "final_answer": "final_answer",
            END: END,
        },
    )

    return g.compile()


# -----------------------------
# Run
# -----------------------------

def main():
    app = build_graph()

    queries = [
        "Provide the details of work order WO1",
        "What work order is scheduled today and how to clean the machine?",
        "How do I clean product PROD-77881?",
    ]

    for q in queries:
        result = app.invoke(
            {
                "user_question": q,
                # optional inputs if user gave IDs directly
                "work_order_id": "WO1" if "WO1" in q else None,
                "product_id": "PROD-77881" if "PROD-77881" in q else None,
            }
        )

        print("\n==============================")
        print("USER:", q)
        print("------------------------------")
        print(result.get("final_answer"))


if __name__ == "__main__":
    main()
