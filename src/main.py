import os

from dotenv import load_dotenv
from openai import AzureOpenAI

_ = load_dotenv()

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
apikey = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# Create OpenAI client
client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=apikey
)

# Test completions
messages = [
    {
        "role": "system",
        "content": "Eres un asistente que trabaja en una agencia de viajes. Responde en español a todo.",
    },
    {
        "role": "user",
        "content": "Voy a hacer un viaje a Dallas, en Texas. Qué debería ver?",
    }
]

response = client.chat.completions.create(
    messages=messages,
    max_completion_tokens=13107,
    temperature=1.0,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    model=deployment
)

print(response.choices[0].message.content)
