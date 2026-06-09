"""Exercise 2: Tools and Knowledge Base.

Complete the tool and knowledge base tasks from the codelab.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from common.llm import get_llm


LEGAL_KNOWLEDGE = [
    {
        "id": "ucc_breach",
        "keywords": ["breach", "contract", "remedies", "damages", "ucc"],
        "text": (
            "Under the Uniform Commercial Code (UCC) Article 2, remedies for breach of contract "
            "include: (1) expectation damages; (2) consequential damages; (3) specific performance; "
            "(4) cover damages. Statute of limitations is typically 4 years (UCC sec. 2-725)."
        ),
    },
    {
        "id": "labor_law",
        "keywords": ["lao dong", "sa thai", "hop dong lao dong", "labor", "termination"],
        "text": (
            "Vietnam's 2019 Labor Code allows unilateral termination only in specific cases, "
            "such as when an employee repeatedly fails to complete work, suffers a prolonged "
            "illness, or when force majeure applies. Termination must follow the statutory "
            "grounds and procedure."
        ),
    },
]


@tool
def search_legal_knowledge(query: str) -> str:
    """Search the legal knowledge base."""
    query_lower = query.lower()
    for entry in LEGAL_KNOWLEDGE:
        if any(kw in query_lower for kw in entry["keywords"]):
            return f"[{entry['id']}] {entry['text']}"
    return "Khong tim thay thong tin lien quan."


@tool
def check_statute_of_limitations(case_type: str) -> str:
    """Check the limitation period for a case type."""
    limits = {
        "contract": "4 nam (UCC sec. 2-725)",
        "tort": "2-3 nam tuy bang",
        "property": "5 nam",
    }
    return limits.get(case_type.lower(), "Khong xac dinh")


async def main():
    load_dotenv()
    llm = get_llm()

    tools = [search_legal_knowledge, check_statute_of_limitations]
    llm_with_tools = llm.bind_tools(tools)

    question = "Thoi hieu khoi kien vu vi pham hop dong la bao lau?"

    messages = [
        SystemMessage(content="Ban la chuyen gia phap ly. Su dung tools de tra cuu thong tin."),
        HumanMessage(content=question),
    ]

    print(f"Cau hoi: {question}\n")

    response = await llm_with_tools.ainvoke(messages)
    messages.append(response)

    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"Tool: {tool_call['name']}")
            tool_result = None

            if tool_call["name"] == "search_legal_knowledge":
                tool_result = search_legal_knowledge.invoke(tool_call["args"])
            elif tool_call["name"] == "check_statute_of_limitations":
                tool_result = check_statute_of_limitations.invoke(tool_call["args"])

            if tool_result:
                messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))

        final_response = await llm_with_tools.ainvoke(messages)
        answer = final_response.content.strip()
        if not answer:
            answer = "\n".join(message.content for message in messages if isinstance(message, ToolMessage))
        print(f"\nKet qua:\n{answer}")
    else:
        print(f"\nKet qua:\n{response.content}")


if __name__ == "__main__":
    asyncio.run(main())
