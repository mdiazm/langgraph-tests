import json
from typing import Literal

from langchain.messages import SystemMessage, ToolMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from src.llm.openai import llm
from src.models.tool_agent import ToolAgentState
from src.tools.date import get_current_date


def first_node(state: ToolAgentState) -> dict:
    """Just a dummy node in an agentic workflow"""
    llm_calls: int = state.llm_calls + 1
    message: AIMessage = AIMessage(content="This is the first message from the fake LLM")

    return {
        "messages": [message],
        "llm_calls": llm_calls
    }


def second_node(state: ToolAgentState) -> dict:
    """Another dummy node that receives the response from the previous LLM and that's it"""
    llm_calls: int = state.llm_calls + 1
    message: AIMessage = AIMessage(content="This is the first message from the fake LLM")

    return {
        "messages": [message],
        "llm_calls": llm_calls
    }


# Define the agent workflow
agent = StateGraph(ToolAgentState)
agent.add_node("first_node", first_node)
agent.add_node("second_node", second_node)

agent.add_edge(START, "first_node")
agent.add_edge("first_node", "second_node")
agent.add_edge("second_node", END)


memory = InMemorySaver()
app = agent.compile(checkpointer=memory)
# Run the agent

if __name__ == '__main__':

    config = {"configurable": {"thread_id": "1"}}
    result = app.invoke({"messages": [HumanMessage("How are you")]}, config)
    result_memory = app.invoke({"messages": [HumanMessage("How are you")]}, config)
    print(result_memory)