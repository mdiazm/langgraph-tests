from datetime import datetime

from langchain.tools import tool


@tool
def get_current_date() -> str:
    """
    A tool to get the current datetime when is called from an agent.

    Returns:
        str: current date formatted as dd-mm-yyyy
    """
    date = datetime.now()

    return date.strftime("%d-%m-%Y")
