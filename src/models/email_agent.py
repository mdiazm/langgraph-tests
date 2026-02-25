from typing import TypedDict, Literal


class EmailClassification(TypedDict):
    """This class will serve only as the output for the email classification"""
    intent: Literal["question", "bug", "billing", "feature", "complex"]
    urgency: Literal["low", "medium", "high", "critical"]
    topic: str
    summary: str


class EmailAgentState(TypedDict):
    """This class implements the email agent state"""
    email_content: str
    sender_email: str
    email_id: str

    # Classification result
    classification: EmailClassification | None

    # Raw search or API results
    search_results: list[str] | None # List of raw document chunks
    customer_history: dict | None # Raw customer data from CRM

    # Generated content
    draft_response: str | None
    messages: list[str] | None