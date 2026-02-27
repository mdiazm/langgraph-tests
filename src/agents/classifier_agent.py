from pprint import pprint
from typing import Literal

from langchain.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from src.llm.openai import llm
from src.models.classifier_agent import ClassifierAgentState, IntentClassification


def classifier_node(state: ClassifierAgentState) -> dict:
    """Classifies the user input into predefined classes"""

    structured_model = llm.with_structured_output(IntentClassification)

    # Prepare the classification prompt
    classification_prompt = SystemMessage(
        content=f"""
            Analyze the user input and classify it:
                
            User input: {state.user_input}
            
            Provide classification on user input.
        """
    )

    intent = structured_model.invoke([classification_prompt])

    return {
        "detected_intent": intent
    }


def generate_recipe(state: ClassifierAgentState) -> dict:
    """Node that generates a recipe to what the user is demanding."""

    prompt = SystemMessage(
        content=f"""
            You are an expert in fine cuisine. You will receive a recipe from a customer and you have to generate
            a plan to prepare that dish. First, enumerate all the ingredients that are needed to prepare the food, 
            include quantities, use grams (g), mililiters (ml) as measures for mass and volumes.
            Prepare a plan explained step by step, don't miss any detail. 
            
            User's request for food: {state.user_input}
            
            Prepare ingredients and plan to cook that dish.
        """
    )

    response = llm.invoke([prompt])

    return {
        "messages": [response.content]
    }


def generate_computer_manual(state: ClassifierAgentState) -> dict:
    """Node that generates a recipe to what the user is demanding."""

    prompt = SystemMessage(
        content=f"""
            You are an expert in customer service to a IT company. You will receive tickets on support requests and
            you have to explain, step by step and based on what you know, how to solve that problem.

            User's request for support: {state.user_input}

            Explain what is the cause of the issue and prepare a step-by-step plan to solve it.
        """
    )

    response = llm.invoke([prompt])

    return {
        "messages": [response.content]
    }


def route_by_request(state: ClassifierAgentState) -> Literal["end", "recipes", "computers"]:
    """Route the flow based on the detected intent"""

    intent = state.detected_intent['intent']

    if intent == "computers":
        return "computers"
    elif intent == "recipes":
        return "recipes"
    else:
        return "end"


# Try out agent
agent = StateGraph(ClassifierAgentState)
agent.add_node("classifier_node", classifier_node)
agent.add_node("generate_recipe", generate_recipe)
agent.add_node("generate_computer_manual", generate_computer_manual)

agent.add_edge(START, "classifier_node")
agent.add_conditional_edges(
    "classifier_node",
    route_by_request,
    {
        "recipes": "generate_recipe",
        "computers": "generate_computer_manual",
        "end": END
    }
)

agent.add_edge("generate_recipe", END)
agent.add_edge("generate_computer_manual", END)

app = agent.compile()

if __name__ == "__main__":
    initial_state = {
        "user_input": "I have a terrible headache"
    }
    result = app.invoke(initial_state)
    pprint(result)