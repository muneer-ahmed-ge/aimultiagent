"""
orchestrator_multi_agent_langgraph.py

A production-friendly skeleton for a multi-agent orchestrator using LangGraph.

Goal:
- Orchestrator agent:
  - Uses an LLM planner to decide which specialist agents to run and in what order
  - Delegates to specialist agents (each specialist is a node; can internally be its own LangGraph)
  - Collects structured outputs (JSON/dicts) into shared state
  - Uses one final LLM call to compose a grounded, well-written response

Notes:
- Specialist agents should return STRUCTURED DATA (dict/JSON), not prose.
- Final composition is the only place you "rephrase" for the user.

You can paste this into PyCharm and run.
"""

from __future__ import annotations

import json
import re
from typing import TypedDict, List, Literal, Optional, Dict, Any

from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END


# --------------------------------------------------------------------
# Types
# --------------------------------------------------------------------

AgentName = Literal["scheduling_agent", "service_insights_agent", "knowledge_agent", "final_answer"]


class OrchestratorState(TypedDict, total=False):
    # Input
    user_question: str

    # Planner output
    plan: List[AgentName]

    # Extracted / shared identifiers (cross-agent dependencies)
    work_order_id: Optional[str]
    product_id: Optional[str]

    # Specialist outputs (structured)
    scheduling_result: Dict[str, Any]
    service_insights_result: Dict[str, Any]
    knowledge_result: Dict[str, Any]

    # Final user response
    final_answer: str


# --------------------------------------------------------------------
# Helpers: robust ID extraction without extra LLM calls
# --------------------------------------------------------------------

WO_REGEX = re.compile(r"\bWO[-_ ]?\d+\b", re.IGNORECASE)
PROD_REGEX = re.compile(r"\bPROD[-_ ]?\d+\b", re.IGNORECASE)


def extract_work_order_id(text: str) -> Optional[str]:
    m = WO_REGEX.search(text)
    return m.group(0).replace(" ", "").replace("_", "-").upper() if m else None


def extract_product_id(text: str) -> Optional[str]:
    m = PROD_REGEX.search(text)
    return m.group(0).replace(" ", "").replace("_", "-").upper() if m else None


# --------------------------------------------------------------------
# 1) Planner node (LLM) => produces plan
# --------------------------------------------------------------------

def planner_node(state: OrchestratorState) -> OrchestratorState:
    """
    LLM decides which specialist agents to run and in what order.

    IMPORTANT:
    - This node does NOT execute tools.
    - It only returns a PLAN (list of agent names).
    """
    model = AzureChatOpenAI(deployment_name="SMAX-AI-Dev-GPT4")

    user_q = state["user_question"]

    prompt = f"""
You are the planner for a ServiceMax multi-agent orchestrator.

Specialist agents available:
- scheduling_agent: returns today's scheduled work order (work_order_id)
- service_insights_agent: returns details/type/history for a work_order_id and includes product_id
- knowledge_agent: returns documentation/cleanup steps for a product_id
- final_answer: must always be last

Rules:
- If user asks what is scheduled today, include scheduling_agent.
- If user asks for work order details/type/history, include service_insights_agent.
- If user asks for cleaning steps/docs, include knowledge_agent.
- Always include final_answer last.
- If the user provides a work_order_id directly, scheduling_agent is optional.
- If the user provides a product_id directly, you may skip service_insights_agent and go to knowledge_agent.

Return ONLY JSON:
{{
  "plan": ["scheduling_agent", "service_insights_agent", "knowledge_agent", "final_answer"]
}}

User question:
{user_q}
""".strip()

    resp = model.invoke(prompt)

    try:
        data = json.loads(resp.content)
        plan = data.get("plan", [])
    except Exception:
        # Safe fallback: always at least attempt final answer
        plan = ["final_answer"]

    # Ensure final_answer last
    if not plan or plan[-1] != "final_answer":
        plan.append("final_answer")

    # Extract IDs from user question WITHOUT extra LLM calls (cheap + robust)
    wo_id = extract_work_order_id(user_q)
    prod_id = extract_product_id(user_q)

    update: OrchestratorState = {"plan": plan}
    if wo_id:
        update["work_order_id"] = wo_id
    if prod_id:
        update["product_id"] = prod_id

    return update


# --------------------------------------------------------------------
# 2) Specialist agent nodes (each can be its own LangGraph)
#    For now we implement them as functions returning structured dicts.
# --------------------------------------------------------------------

def scheduling_agent_node(state: OrchestratorState) -> OrchestratorState:
    """
    Specialist agent: Scheduling
    - In real life: can be its own LangGraph with tools (calendar, dispatcher, FSM, etc.)
    - Returns structured output
    """
    # MOCK / replace with your scheduling APIs
    result = {
        "agent": "scheduling_agent",
        "date": "today",
        "technician_id": "TECH-42",
        "work_order_id": "WO-100245",
        "scheduled_time": "10:00 AM",
        "site": "Acme Plant - San Jose",
    }

    return {
        "scheduling_result": result,
        "work_order_id": result["work_order_id"],
    }


def service_insights_agent_node(state: OrchestratorState) -> OrchestratorState:
    """
    Specialist agent: Service Insights
    Depends on work_order_id.
    Returns structured output including product_id.
    """
    work_order_id = state.get("work_order_id")
    if not work_order_id:
        # Graph-safe fallback
        result = {
            "agent": "service_insights_agent",
            "error": "Missing work_order_id",
        }
        return {"service_insights_result": result}

    # MOCK / replace with your ServiceMax APIs / SFDC queries
    result = {
        "agent": "service_insights_agent",
        "work_order_id": work_order_id,
        "work_order_type": "Critical",
        "product_id": "PROD-77881",
        "symptom_summary": "Oil leakage near primary valve",
        "last_actions": [
            "2025-12-02: Replaced seal kit",
            "2025-10-18: Inspection and lubrication",
        ],
        "priority": "High",
    }

    return {
        "service_insights_result": result,
        "product_id": result["product_id"],
    }


def knowledge_agent_node(state: OrchestratorState) -> OrchestratorState:
    """
    Specialist agent: Knowledge / Docs
    Depends on product_id.
    Returns structured doc steps.
    """
    product_id = state.get("product_id")
    if not product_id:
        result = {
            "agent": "knowledge_agent",
            "error": "Missing product_id",
        }
        return {"knowledge_result": result}

    # MOCK / replace with your AIKA_ANSWER / KB search / RAG
    result = {
        "agent": "knowledge_agent",
        "product_id": product_id,
        "doc_title": "Cleaning & Maintenance Guide - Hydraulic Press Series",
        "cleanup_steps": [
            "Power down and lockout/tagout before cleaning.",
            "Wipe exterior surfaces using a non-abrasive cloth.",
            "Use approved degreaser on oil residue near the valve housing.",
            "Inspect seals and fittings after cleaning for leak recurrence.",
            "Run a short test cycle and confirm pressure stability.",
        ],
        "source": "KnowledgeBase",
    }

    return {"knowledge_result": result}


# --------------------------------------------------------------------
# 3) Final answer composer (LLM) => rephrase + merge outputs for the user
# --------------------------------------------------------------------

def final_answer_node(state: OrchestratorState) -> OrchestratorState:
    """
    This is the ONLY place you should do "rephrasing" for user output.
    It takes structured outputs and converts them into a clean response.
    """
    model = AzureChatOpenAI(deployment_name="SMAX-AI-Dev-GPT4")

    prompt = f"""
You are a ServiceMax field service assistant.
Only use the factual data provided below. Do not invent details.

User question:
{state.get("user_question")}

Scheduling result (if available):
{json.dumps(state.get("scheduling_result", {}), indent=2)}

Service insights result (if available):
{json.dumps(state.get("service_insights_result", {}), indent=2)}

Knowledge result (if available):
{json.dumps(state.get("knowledge_result", {}), indent=2)}

Write a concise final answer that:
- Answers the user's question directly
- Includes: work_order_id, work_order_type (if available)
- Includes cleanup steps if available (as bullets)
- If something is missing (ex: product_id), say that clearly
""".strip()

    resp = model.invoke(prompt)
    return {"final_answer": resp.content}


# --------------------------------------------------------------------
# Loop control: execute plan step-by-step
# --------------------------------------------------------------------

def next_step_router(state: OrchestratorState) -> str:
    """
    Chooses the next agent node to run based on the current plan.
    """
    plan = state.get("plan", [])
    if not plan:
        return END
    return plan[0]


def pop_step_node(state: OrchestratorState) -> OrchestratorState:
    plan = state.get("plan", [])
    if plan:
        plan = plan[1:]
    return {"plan": plan}


# --------------------------------------------------------------------
# Build orchestrator graph
# --------------------------------------------------------------------

def build_orchestrator_graph():
    g = StateGraph(OrchestratorState)

    # Nodes
    g.add_node("planner", planner_node)

    g.add_node("scheduling_agent", scheduling_agent_node)
    g.add_node("service_insights_agent", service_insights_agent_node)
    g.add_node("knowledge_agent", knowledge_agent_node)
    g.add_node("final_answer", final_answer_node)

    g.add_node("pop_step", pop_step_node)

    # Entry
    g.set_entry_point("planner")

    # After planner, choose first step
    g.add_conditional_edges(
        "planner",
        next_step_router,
        {
            "scheduling_agent": "scheduling_agent",
            "service_insights_agent": "service_insights_agent",
            "knowledge_agent": "knowledge_agent",
            "final_answer": "final_answer",
            END: END,
        },
    )

    # After each specialist agent, pop step, then decide again
    for step in ["scheduling_agent", "service_insights_agent", "knowledge_agent", "final_answer"]:
        g.add_edge(step, "pop_step")

    g.add_conditional_edges(
        "pop_step",
        next_step_router,
        {
            "scheduling_agent": "scheduling_agent",
            "service_insights_agent": "service_insights_agent",
            "knowledge_agent": "knowledge_agent",
            "final_answer": "final_answer",
            END: END,
        },
    )

    return g.compile()


# --------------------------------------------------------------------
# Demo run
# --------------------------------------------------------------------

def main():
    app = build_orchestrator_graph()

    queries = [
        "What work order is scheduled today? Tell me details and how to clean it.",
        "Provide details of work order WO1",
        "How do I clean product PROD-77881?",
    ]

    for q in queries:
        result = app.invoke({"user_question": q})
        print("\n=================================================")
        print("USER:", q)
        print("-------------------------------------------------")
        print(result.get("final_answer", "No final_answer produced."))


if __name__ == "__main__":
    main()
