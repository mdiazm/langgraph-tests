# AzureAI - LangGraph Testing Repository

This repository is dedicated to testing and experimenting with various capabilities of the [LangGraph](https://langchain-ai.github.io/langgraph.py/) library.

## Overview

LangGraph is a library for building stateful, multi-actor applications with LLMs, built on top of LangChain. This repository serves as a testing ground for exploring different features and patterns available in LangGraph.

## Features Being Tested

- State management with custom state classes
- Interrupt and resume patterns for human-in-the-loop workflows
- Conditional routing and branching logic
- Persistent checkpoints and memory savers
- Integration with Azure OpenAI services
- Multi-agent collaboration patterns

## Project Structure

```
src/
├── agents/          # Different agent implementations
├── models/          # State and data models
└── main.py         # Entry points and examples
docs/               # Documentation and explanations
data/               # Sample data for testing
```

## Key Components

### Email Agent
A primary example demonstrating:
- Email classification and routing
- Human review interrupt mechanisms
- Dynamic workflow routing with Commands
- Integration with Azure OpenAI

### Basic Agent
A simpler example for foundational concepts.

## Setup

1. Install dependencies using uv:
   ```bash
   uv sync
   ```

2. Configure your environment variables in `.env`:
   ```
   AZURE_OPENAI_ENDPOINT=your_endpoint
   AZURE_OPENAI_API_KEY=your_api_key
   AZURE_OPENAI_DEPLOYMENT=your_deployment
   AZURE_OPENAI_API_VERSION=your_api_version
   AZURE_OPENAI_MODEL=your_model
   ```

## Documentation

See the `docs/` folder for detailed explanations of implemented patterns and concepts.

## Contributing

This is primarily a personal testing repository, but suggestions and improvements are welcome.