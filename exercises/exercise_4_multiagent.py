"""Exercise 4: Add a Privacy Agent to the Multi-Agent System."""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Annotated, TypedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from common.llm import get_llm


def _last_wins(left: str | None, right: str | None) -> str:
    """Reducer: latest non-empty value wins."""
    return right if right is not None else (left or "")


class State(TypedDict):
    question: str
    law_analysis: Annotated[str, _last_wins]
    tax_analysis: Annotated[str, _last_wins]
    compliance_analysis: Annotated[str, _last_wins]
    privacy_analysis: Annotated[str, _last_wins]
    final_response: str
    needs_tax: bool
    needs_compliance: bool
    needs_privacy: bool


def law_agent(state: State) -> dict:
    """General legal analysis."""
    llm = get_llm()
    prompt = f"""Ban la chuyen gia phap ly. Phan tich cau hoi sau:

{state['question']}

Tap trung vao: hop dong, trach nhiem dan su, quyen va nghia vu phap ly."""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"law_analysis": response.content}


def check_routing(state: State) -> dict:
    """Decide which specialist agents are needed."""
    question_lower = state["question"].lower()
    return {
        "needs_tax": any(kw in question_lower for kw in ["tax", "irs", "thue"]),
        "needs_compliance": any(kw in question_lower for kw in ["compliance", "sec", "regulation", "aml"]),
        "needs_privacy": any(kw in question_lower for kw in ["data", "privacy", "gdpr", "du lieu", "breach"]),
    }


def route_to_specialists(state: State) -> list[Send]:
    """Dispatch specialist nodes in parallel."""
    sends: list[Send] = []
    if state.get("needs_tax"):
        sends.append(Send("tax_agent", state))
    if state.get("needs_compliance"):
        sends.append(Send("compliance_agent", state))
    if state.get("needs_privacy"):
        sends.append(Send("privacy_agent", state))
    if not sends:
        sends.append(Send("aggregate_results", state))
    return sends


def tax_agent(state: State) -> dict:
    """Tax specialist."""
    llm = get_llm()
    prompt = f"""Ban la chuyen gia thue. Phan tich khia canh thue trong cau hoi:

Cau hoi: {state['question']}
Phan tich phap ly: {state.get('law_analysis', 'N/A')}

Tap trung: IRS, tax evasion, penalties, FBAR, FATCA."""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"tax_analysis": response.content}


def compliance_agent(state: State) -> dict:
    """Compliance specialist."""
    llm = get_llm()
    prompt = f"""Ban la chuyen gia compliance. Phan tich khia canh tuan thu:

Cau hoi: {state['question']}
Phan tich phap ly: {state.get('law_analysis', 'N/A')}

Tap trung: SEC, SOX, FCPA, AML, regulatory violations."""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"compliance_analysis": response.content}


def privacy_agent(state: State) -> dict:
    """Privacy and data protection specialist."""
    llm = get_llm()
    prompt = f"""Ban la chuyen gia ve GDPR va bao ve du lieu ca nhan.

Cau hoi: {state['question']}
Phan tich phap ly: {state.get('law_analysis', 'N/A')}

Tap trung: GDPR, data protection, privacy rights, data breach, consent."""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"privacy_analysis": response.content}


def aggregate_results(state: State) -> dict:
    """Combine all analyses into one final report."""
    llm = get_llm()

    sections = []
    if state.get("law_analysis"):
        sections.append(f"PHAN TICH PHAP LY:\n{state['law_analysis']}")
    if state.get("tax_analysis"):
        sections.append(f"PHAN TICH THUE:\n{state['tax_analysis']}")
    if state.get("compliance_analysis"):
        sections.append(f"PHAN TICH TUAN THU:\n{state['compliance_analysis']}")
    if state.get("privacy_analysis"):
        sections.append(f"PHAN TICH PRIVACY:\n{state['privacy_analysis']}")

    combined = "\n\n".join(sections)
    prompt = f"""Tong hop cac phan tich sau thanh mot bao cao phap ly hoan chinh:

{combined}

Cau hoi goc: {state['question']}

Hay tao mot bao cao ngan gon, co cau truc ro rang."""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"final_response": response.content}


def build_graph() -> StateGraph:
    """Build the multi-agent graph."""
    graph = StateGraph(State)

    graph.add_node("law_agent", law_agent)
    graph.add_node("check_routing", check_routing)
    graph.add_node("tax_agent", tax_agent)
    graph.add_node("compliance_agent", compliance_agent)
    graph.add_node("privacy_agent", privacy_agent)
    graph.add_node("aggregate_results", aggregate_results)

    graph.add_edge(START, "law_agent")
    graph.add_edge("law_agent", "check_routing")
    graph.add_conditional_edges(
        "check_routing",
        route_to_specialists,
        ["tax_agent", "compliance_agent", "privacy_agent", "aggregate_results"],
    )
    graph.add_edge("tax_agent", "aggregate_results")
    graph.add_edge("compliance_agent", "aggregate_results")
    graph.add_edge("privacy_agent", "aggregate_results")
    graph.add_edge("aggregate_results", END)

    return graph.compile()


async def main():
    load_dotenv()

    question = "Neu cong ty bi ro ri du lieu khach hang, hau qua phap ly va thue la gi?"

    print("=" * 70)
    print("MULTI-AGENT SYSTEM voi Privacy Agent")
    print("=" * 70)
    print(f"\nCau hoi: {question}\n")
    print("Dang xu ly qua cac agents...\n")

    graph = build_graph()
    result = await graph.ainvoke(
        {
            "question": question,
            "law_analysis": "",
            "tax_analysis": "",
            "compliance_analysis": "",
            "privacy_analysis": "",
            "final_response": "",
            "needs_tax": False,
            "needs_compliance": False,
            "needs_privacy": False,
        }
    )

    print("\n" + "=" * 70)
    print("KET QUA CUOI CUNG")
    print("=" * 70)
    print(result["final_response"])
    print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
