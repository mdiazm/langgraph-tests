from datetime import datetime

from langchain.tools import tool


@tool
def get_current_date() -> str:
    """
    A tool to get the current date when is called from an agent.

    Returns:
        str: current date formatted as dd-mm-YYYY
    """
    date = datetime.now()

    return date.strftime("%d-%m-%Y")

@tool
def get_current_hour() -> str:
    """
    A tool to get the current time when is called from an agent.

    Returns:
        str: current date formatted as HH:MM:SS
    """
    date = datetime.now()

    return date.strftime("%H:%M:%S")
