from __future__ import annotations

from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_openai import AzureChatOpenAI


@tool
def scheduling_service() -> str:
    """Get scheduled work order today."""
    return "work_order_id=WO-100245"


@tool
def service_insights_service(work_order_id: str) -> str:
    """Get product_id from work_order_id."""
    return f"work_order_type=Critical,product_id=PROD-77881 for {work_order_id}"


@tool
def knowledge_access_service(product_id: str) -> str:
    """Get docs / cleanup steps from product_id."""
    return f"cleanup steps for {product_id}"


tools = [
    scheduling_service,
    service_insights_service,
    knowledge_access_service,
]


SYSTEM_PROMPT = """
You are a ServiceMax field service assistant.

Your job:
- Identify the user's goal
- Call tools in the correct order
- Do not hallucinate data
- Use tools whenever information is required
- Produce a concise final answer

Tool dependency rule:
- First call scheduling_service to get the work_order_id
- Then call service_insights_service(work_order_id) to get product_id
- Then call knowledge_access_service(product_id) to get cleanup steps

Final response MUST include:
- work_order_id
- work_order_type
- product_id
- cleanup_steps (bulleted list)

Don't hallucinate just be grounded to the data provided.
"""


def main():
    user_question = "What work order is scheduled today tell me its id and type and how to clean the machine?"

    model = AzureChatOpenAI(deployment_name="SMAX-AI-Dev-GPT4")

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )

    result = agent.invoke({"input": user_question})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
