from pydantic import BaseModel, Field
from typing import List, Annotated
import operator

from langchain.messages import AnyMessage


class ToolAgentState(BaseModel):
    """State for the decision‑only tool agent."""
    messages: Annotated[list[AnyMessage], operator.add] = Field(default_factory=list, description="Chat history")
    llm_calls: int = Field(0, description="Number of LLM invocations")
    tool_needed: bool = Field(False, description="Whether a tool call is required")