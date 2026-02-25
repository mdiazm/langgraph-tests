import os
from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command, RetryPolicy
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import AzureChatOpenAI
from langchain.messages import HumanMessage
from dotenv import load_dotenv

from src.models.email_agent import EmailAgentState, EmailClassification

_ = load_dotenv()

deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
model = os.getenv("AZURE_OPENAI_MODEL")

# Create OpenAI client
llm = AzureChatOpenAI(
    api_version=api_version,
    azure_deployment=deployment,
    model=model,
    temperature=0
)


def read_email(state: EmailAgentState) -> dict:
    """Extract and parse email content"""
    # In a production environment, this would connect to email service
    return {
        "messages": [HumanMessage(content=f"Processing email {state['email_content']}")]
    }


def classify_intent(state: EmailAgentState) -> Command[Literal[
    "search_documentation", "human_review", "draft_response", "bug_tracking"
]]:
    """Use LLM to classify email intent and urgency, then route accordingly"""

    # Create structured LLM that returns EmailClassification dict
    # This gives information to the LLM to output the same structure that the sent one
    structured_llm = llm.with_structured_output(EmailClassification)

    # Format the prompt on demand, not the stored in the state
    classification_prompt = f"""
    Analyze this customer email and classifiy it:
    
    Email: {state['email_content']}
    From: {state['sender_email']}
    
    Provide classification including intent, urgency, topic, and summary.
    """

    # Get structured response directly as dict
    classification = structured_llm.invoke(classification_prompt)

    # Determine next node based on classification (with the LLM)
    intent = classification['intent']
    urgency = classification['urgency']

    if intent == 'billing' or urgency == 'critical':
        goto = "human_review"
    elif intent in ['question', 'feature']:
        goto = "search_documentation"
    elif intent == 'bug':
        goto = "bug_tracking"
    else:
        goto = "draft_response"

    # Store classification as a single dict in state
    return Command(
        update={"classification": classification},
        goto=goto
    )


# Search and tracking nodes
def search_documentation(state: EmailAgentState) -> Command[Literal[
    "draft_response"
]]:
    """Search knowledge base for relevant information"""

    # Build search query from classification
    classification = state.get('classification', {})
    query = f"{classification.get('intent', '')} {classification.get('topic', '')}"

    try:
        # Placeholder hardcoded search results based on intent/topic
        search_results = [
            "Reset password via Settings > Security > Change Password",
            "Password must be at least 12 characters",
            "Include uppercase, lowercase, numbers and symbols",
            "Two‑factor authentication can be enabled in Security settings"
        ]
    except Exception as e:
        search_results = [f"Search temporarily unavailable: {str(e)}"]

    return Command(
        update={"search_results": search_results},
        goto="draft_response"
    )


def bug_tracking(state: EmailAgentState) -> Command[Literal["draft_response"]]:
    """Create or update bug tracking ticket"""

    # Create ticket in your bug tracking system
    ticket_id = "BUG-12345"  # This is created through API

    return Command(
        update={
            "search_results": [f"Bug ticket {ticket_id} created"],
            "current_step": "bug_tracked"
        },
        goto="draft_response"
    )


# Response nodes
def draft_response(state: EmailAgentState) -> Command[Literal["human_review", "send_reply"]]:
    """Generate response using context and route based on quality"""

    classification = state.get('classification', {})

    # Format context from raw state data on-demand
    context_sections = []

    if state.get('search_results'):
        # Format search results for the prompt
        formatted_docs = "\n".join([f"- {doc}" for doc in state['search_results']])
        context_sections.append(f"Relevant documentation:\n{formatted_docs}")

    if state.get('customer_history'):
        # Format customer data for the prompt
        context_sections.append(f"Customer tier: {state['customer_history'].get('tier', 'standard')}")

    # Build the prompt with formatted context
    draft_prompt = f"""
    Draft a response to this customer email:
    {state['email_content']}
    
    Email intent: {classification.get('intent', 'unknown')}
    Urgency level: {classification.get('urgency', 'medium')}
    
    {chr(10).join(context_sections)}
    
    Guidelines:
    - Be professional and helpful
    - Address their specific concern
    - Use the provided documentation when relevant
    - Use my name "Miguel Díaz Medina" to close the email, but don't let any template to fill manually.
    """

    response = llm.invoke(draft_prompt)

    # Determine if human review needed based on urgency and intent
    needs_review = (
        classification.get('urgency') in ['critical'] or
        classification.get('intent') == 'complex'
    )

    # Route to appropriate next node
    goto = "human_review" if needs_review else "send_reply"

    return Command(
        update={"draft_response": response.content},  # Store only raw response
        goto=goto
    )


def human_review(state: EmailAgentState) -> Command[Literal["send_reply", END]]:
    """Pause for human review using interrupt and route based on decision"""

    classification = state.get('classification', {})

    # interrupt() must come first - any code before it will re-run on resume
    human_decision = interrupt({
        "email_id": state.get('email_id', ''),
        "original_email": state.get('email_content', ''),
        "draft_response": state.get('draft_response', ''),
        "urgency": classification.get('urgency'),
        'intent': classification.get('intent'),
        "action": "Please review and approve/edit this response"
    })

    # Now process the human's decision
    if human_decision.get("approved"):
        return Command(
            update={"draft_response": human_decision.get("edited_response", state.get('draft_response', ''))},
            goto="send_reply"
        )
    else:
        # Rejection means human will handle directly
        return Command(update={}, goto=END)


def send_reply(state: EmailAgentState) -> dict:
    """Send the email response"""
    # TODO: integrate with email service
    print(f"Sending reply: {state['draft_response']}")
    return {}


# Compile the agent. Since we are using Command, we don't need to define each edge
workflow = StateGraph(EmailAgentState)

# Add nodes with appropriate error handling
workflow.add_node("read_email", read_email)
workflow.add_node("classify_intent", classify_intent)

# Add retry policy for nodes that might have transient failures
workflow.add_node(
    "search_documentation",
    search_documentation,
    retry_policy=RetryPolicy(max_attempts=5)
)
workflow.add_node("bug_tracking", bug_tracking)
workflow.add_node("draft_response", draft_response)
workflow.add_node("human_review", human_review)
workflow.add_node("send_reply", send_reply)

# Add only the essential edges (those that cannot be routed with Commands)
workflow.add_edge(START, "read_email")
workflow.add_edge("read_email", "classify_intent")
workflow.add_edge("send_reply", END)

# Compile with checkpointer for persistence
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


# Test the agent
initial_state = {
    "email_content": "I was double charged with the same topic, give me my money!!!!",
    "sender_email": "customer@example.com",
    "email_id": "email_123",
    "messages": []
}

config = {"configurable": {"thread_id": "customer_123"}}
result = app.invoke(initial_state, config)

print(f"human review interrupt: {result['__interrupt__']}")

human_response = Command(
    resume={
        "approved": True,
        "edited_response": "We sincerely apologize for the double charge..."
    }
)

final_result = app.invoke(human_response, config)
print(f"Email sent successfully")