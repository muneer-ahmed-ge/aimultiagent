from __future__ import annotations

from typing import TypedDict, Literal
from typing_extensions import Annotated

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


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


TOOLS = [scheduling_service, service_insights_service, knowledge_access_service]


SYSTEM_PROMPT = """
You are a ServiceMax field service assistant.

Rules:
- Use tools in correct order when needed:
  1) scheduling_service -> work_order_id
  2) service_insights_service(work_order_id) -> type, product_id
  3) knowledge_access_service(product_id) -> cleanup steps
- Don't hallucinate; only use tool results.
- Stop when you can answer fully.
""".strip()


# -----------------------------
# State with message reducer ✅
# -----------------------------

class GraphState(TypedDict):
    # add_messages is the reducer that APPENDS messages across nodes
    messages: Annotated[list[BaseMessage], add_messages]


# -----------------------------
# Brain node (LLM decides tool vs final)
# -----------------------------

def brain_node(state: GraphState) -> GraphState:
    model = AzureChatOpenAI(deployment_name="SMAX-AI-Dev-GPT4")
    model_with_tools = model.bind_tools(TOOLS)

    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}  # reducer will append


# -----------------------------
# Continue routing
# -----------------------------

def should_continue(state: GraphState) -> Literal["tools", "__end__"]:
    last_msg = state["messages"][-1]
    tool_calls = getattr(last_msg, "tool_calls", None)
    if tool_calls:
        return "tools"
    return END


# -----------------------------
# Build graph
# -----------------------------

def build_graph():
    g = StateGraph(GraphState)

    g.add_node("brain", brain_node)
    g.add_node("tools", ToolNode(TOOLS))

    g.set_entry_point("brain")

    g.add_conditional_edges(
        "brain",
        should_continue,
        {
            "tools": "tools",
            END: END,
        },
    )

    # tool execution -> back to brain (ReAct loop)
    g.add_edge("tools", "brain")

    return g.compile()


# -----------------------------
# Run
# -----------------------------

def main():
    app = build_graph()

    user_question = "What work order is scheduled today tell me its id and type and how to clean the machine?"

    result = app.invoke(
        {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_question),
            ]
        }
    )

    print("\n--- FINAL ANSWER ---\n")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
