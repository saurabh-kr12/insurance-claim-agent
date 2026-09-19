from typing import Annotated, TypedDict, List, Literal

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage

from src.config import get_llm
from src.agents.tools import (
    retrieve_claim_documents,
    summarize_claim,
    check_missing_information,
    flag_fraud_risk,
    evaluate_hallucination_tool,
    test_bias_tool,
)

WORKER_NAMES = ["Retriever", "Summarizer", "FraudRisk", "Auditor"]

WORKER_TOOLS = {
    "Retriever": [retrieve_claim_documents],
    "Summarizer": [summarize_claim],
    "FraudRisk": [flag_fraud_risk, check_missing_information],
    "Auditor": [evaluate_hallucination_tool, test_bias_tool],
}

WORKER_SYSTEM_PROMPTS = {
    "Retriever": (
        "You are the Retriever specialist. Your only job is to find relevant claim "
        "document excerpts using the retrieve_claim_documents tool and report what "
        "you found, with citations. Do not summarize or judge the claim -- just "
        "retrieve and report."
    ),
    "Summarizer": (
        "You are the Summarizer specialist. Your only job is to produce a clear, "
        "concise summary of a claim using the summarize_claim tool."
    ),
    "FraudRisk": (
        "You are the FraudRisk specialist. Your job is to check for missing "
        "information and potential fraud risk indicators using your tools. Always "
        "phrase findings as indicators for human review, never as accusations."
    ),
    "Auditor": (
        "You are the Auditor specialist (LLMOps quality control). Your job is to "
        "run hallucination checks and bias tests on prior answers using your tools, "
        "and report the verdicts plainly."
    ),
}


class SupervisorDecision(BaseModel):
    """Structured output the supervisor LLM must produce each turn."""
    next: Literal["Retriever", "Summarizer", "FraudRisk", "Auditor", "FINISH"] = Field(
        description="Which specialist should act next, or FINISH if the task is complete."
    )
    reason: str = Field(description="One sentence explaining why this choice was made.")


class MultiAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    next_worker: str


SUPERVISOR_PROMPT = """You are the Supervisor coordinating a team of claims-review \
specialists:
- Retriever: finds relevant claim document excerpts.
- Summarizer: summarizes a claim.
- FraudRisk: checks for missing info and fraud risk indicators.
- Auditor: runs hallucination checks and bias tests on prior answers.

Given the conversation so far, decide which specialist should act next to make \
progress on the user's request, or respond FINISH if the request has already been \
fully addressed by the specialists' outputs above.

Only pick specialists that are actually relevant to what the user asked. For a \
simple summary request, Retriever then Summarizer may be enough -- you do not need \
to invoke every specialist for every request.
"""


def supervisor_node(state: MultiAgentState) -> dict:
    """Decides which specialist acts next using structured output."""
    llm = get_llm()
    structured_llm = llm.with_structured_output(SupervisorDecision)

    full_messages = [SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"]
    decision: SupervisorDecision = structured_llm.invoke(full_messages)

    routing_note = AIMessage(
        content=f"[Supervisor] -> {decision.next} ({decision.reason})"
    )
    return {"messages": [routing_note], "next_worker": decision.next}


def make_worker_node(worker_name: str):
    """Factory that builds a graph node function for a given specialist."""

    def worker_node(state: MultiAgentState) -> dict:
        llm = get_llm().bind_tools(WORKER_TOOLS[worker_name])
        system = SystemMessage(content=WORKER_SYSTEM_PROMPTS[worker_name])
        # Give the worker the full conversation so it has context on the
        # original user request plus what other specialists already found.
        response = llm.invoke([system] + state["messages"])

        # If the worker requested a tool call, execute it directly here
        # (kept simple/synchronous rather than adding a separate ToolNode
        # per worker, since each worker only ever needs 1-2 tool calls).
        messages_out: List[BaseMessage] = [response]
        if getattr(response, "tool_calls", None):
            tool_lookup = {t.name: t for t in WORKER_TOOLS[worker_name]}
            for call in response.tool_calls:
                tool_fn = tool_lookup.get(call["name"])
                if tool_fn is None:
                    continue
                result = tool_fn.invoke(call["args"])
                from langchain_core.messages import ToolMessage
                messages_out.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=call["id"],
                        name=call["name"],
                    )
                )
            # One more LLM call so the worker can phrase a final answer using
            # the tool result, tagged with its specialist name for the trace.
            follow_up = llm.invoke([system] + state["messages"] + messages_out)
            follow_up.content = f"[{worker_name}] {follow_up.content}"
            messages_out.append(follow_up)
        else:
            response.content = f"[{worker_name}] {response.content}"

        return {"messages": messages_out}

    return worker_node


def route_from_supervisor(state: MultiAgentState) -> str:
    """Reads state['next_worker'] set by supervisor_node to pick the edge."""
    if state["next_worker"] == "FINISH":
        return END
    return state["next_worker"]


def build_multi_agent_graph():
    graph = StateGraph(MultiAgentState)

    graph.add_node("supervisor", supervisor_node)
    for worker_name in WORKER_NAMES:
        graph.add_node(worker_name, make_worker_node(worker_name))
        graph.add_edge(worker_name, "supervisor")  # workers report back to supervisor

    graph.set_entry_point("supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {**{name: name for name in WORKER_NAMES}, END: END},
    )

    return graph.compile()


_compiled_multi_agent = None


def get_multi_agent():
    global _compiled_multi_agent
    if _compiled_multi_agent is None:
        _compiled_multi_agent = build_multi_agent_graph()
    return _compiled_multi_agent


def run_multi_agent(user_message: str, max_steps: int = 8) -> dict:
    """
    Runs the supervisor graph and returns a trace of which specialists were
    invoked plus a final synthesized answer (the last substantive worker
    message).
    """
    graph = get_multi_agent()
    result = graph.invoke(
        {"messages": [HumanMessage(content=user_message)], "next_worker": ""},
        config={"recursion_limit": max_steps * 2 + 2},
    )

    messages = result["messages"]
    trace = [m.content for m in messages if getattr(m, "content", "")]

    # The final answer is the last worker-tagged message (not a [Supervisor] line).
    final_answer = ""
    for m in reversed(messages):
        content = getattr(m, "content", "")
        if content and not content.startswith("[Supervisor]"):
            final_answer = content
            break

    return {"trace": trace, "final_answer": final_answer}


if __name__ == "__main__":
    # Manual test: python -m src.agents.multi_agent
    output = run_multi_agent("Give me a summary of CLM-1007 and check it for fraud risk.")
    print("=== Trace ===")
    for step in output["trace"]:
        print(step, "\n")
    print("=== Final Answer ===")
    print(output["final_answer"])
