import os

from dotenv import load_dotenv
from langchain_ollama.chat_models import ChatOllama

_ = load_dotenv()

# Get Ollama configuration from environment variables
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1")

# Create Ollama client
llm = ChatOllama(
    model=ollama_model,
    temperature=0.0
)

__all__ = [llm]