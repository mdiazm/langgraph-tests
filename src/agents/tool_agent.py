from pprint import pprint
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END

try:
    from src.llm.openai import llm
except:
    from src.llm.ollama import llm

from src.models.tool_agent import ToolAgentState
from src.tools.date import get_current_date, get_current_hour

# Bind tools to the model
model_with_tools = llm.bind_tools(tools=[get_current_date, get_current_hour])
tools_by_name = {"get_current_date": get_current_date, "get_current_hour": get_current_hour}

# Node definition

def llm_call(state: ToolAgentState) -> dict:
    """Perform a call to LLM to decide whether a tool is needed."""
    system: SystemMessage = SystemMessage(content="You are a helpful assistant. Talk like if you were a pirate")
    response = model_with_tools.invoke([system]+ state.messages)

    return {
        "messages": [response]
    }

def tool_node(state: ToolAgentState) -> dict:
    """This node evals if any tool needs to be called and, in that case, it executes the tool"""

    result = []
    for tool_call in state.messages[-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))

    return {
        "messages": result
    }

def should_continue(state: ToolAgentState) -> Literal["tool_node", END]:
    """Decide whether the tool node must be called"""

    messages = state.messages
    last_message = messages[-1]

    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, reply to the user
    return END


# Agent structure
agent = StateGraph(ToolAgentState)

agent.add_node("llm_call", llm_call)
agent.add_node("tool_node", tool_node)

agent.add_edge(START, "llm_call")
agent.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent.add_edge("tool_node", "llm_call")

app = agent.compile()

if __name__ == "__main__":
    result = app.invoke({"messages": [HumanMessage(content="What time is it? Which month are we on?")]})

    pprint(result)