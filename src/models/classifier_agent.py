import operator
from pydantic import BaseModel, Field
from typing import List, Annotated, Literal, TypedDict

from langchain.messages import AnyMessage


class IntentClassification(TypedDict):
    intent: Literal["computers", "recipes", "others"]


class ClassifierAgentState(BaseModel):
    messages: Annotated[List[AnyMessage], operator.add] = Field("The list of messages sent to the LLM")
    detected_intent: IntentClassification | None = Field("Detected intent in the user input")
    user_input: str = Field("What the user is requesting to the assistant. This is not related to memory/LLM")

