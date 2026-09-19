from typing import Annotated, TypedDict, List

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

from src.config import get_llm
from src.agents.tools import ALL_TOOLS
from src.agents.prompts import AGENT_SYSTEM_PROMPT


class AgentState(TypedDict):
    """
    The shared state that flows through every node in the graph.
    `messages` accumulates the full conversation (including tool calls and
    their results) using LangGraph's add_messages reducer, which appends
    new messages instead of overwriting the list.
    """
    messages: Annotated[List[BaseMessage], add_messages]


def _build_llm_with_tools():
    """Binds our tool list to the LLM so it can emit tool-call requests."""
    llm = get_llm()
    return llm.bind_tools(ALL_TOOLS)


def agent_node(state: AgentState) -> dict:
    """
    The "brain" node: given the conversation so far, decides whether to
    respond directly or call a tool. Returns the new message to append.
    """
    llm_with_tools = _build_llm_with_tools()
    messages = state["messages"]

    # Ensure the system prompt is always present at the start of the
    # conversation sent to the LLM (but we don't persist it into `messages`
    # every turn -- we prepend it fresh each call).
    full_messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)] + messages
    response = llm_with_tools.invoke(full_messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """
    Routing function: looks at the last message. If the LLM requested a
    tool call, route to the "tools" node. Otherwise, we're done -> END.
    """
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


def build_agent_graph():
    """
    Assembles the graph:
      START -> agent -> (conditional) -> tools -> agent -> ... -> END
    """
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")  # after running a tool, go back to the agent

    return graph.compile()


# Build once at import time so Streamlit doesn't recompile it on every rerun.
_compiled_graph = None


def get_agent():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_agent_graph()
    return _compiled_graph


def run_agent(user_message: str) -> dict:
    """
    Runs the agent on a single user message and returns:
      - "final_answer": the agent's final text response
      - "trace": a list of readable strings describing each step taken
        (tool calls + tool results), useful for the "Agent Workflow" UI tab.
    """
    agent = get_agent()
    result = agent.invoke({"messages": [HumanMessage(content=user_message)]})

    messages = result["messages"]
    trace = []
    final_answer = ""

    for msg in messages:
        msg_type = type(msg).__name__
        if msg_type == "AIMessage":
            if getattr(msg, "tool_calls", None):
                for call in msg.tool_calls:
                    trace.append(f"🔧 Agent called tool `{call['name']}` with args {call['args']}")
            if msg.content:
                final_answer = msg.content
                trace.append(f"🤖 Agent response: {msg.content}")
        elif msg_type == "ToolMessage":
            preview = str(msg.content)[:400]
            trace.append(f"📄 Tool `{msg.name}` returned: {preview}")

    return {"final_answer": final_answer, "trace": trace}


if __name__ == "__main__":
    # Manual test: python -m src.agents.graph
    # (Requires the vector store to already be built via `python -m src.vectorstore`)
    output = run_agent("Summarize claim CLM-1006 and flag any fraud risk indicators.")
    print("=== Trace ===")
    for step in output["trace"]:
        print(step)
    print("\n=== Final Answer ===")
    print(output["final_answer"])
