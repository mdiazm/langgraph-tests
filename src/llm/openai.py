import os

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

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

__all__ = [llm]