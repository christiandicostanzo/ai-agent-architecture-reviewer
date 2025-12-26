# AI Agent Application

This is a simple AI Agent application built using LangChain.

## Prerequisites

- Python 3.8+
- OpenAI API Key

## Setup

1.  **Clone the repository** (if you haven't already).
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Configure Environment Variables**:
    - Copy `.env.example` to `.env`:
        ```bash
        cp .env.example .env
        # On Windows Command Prompt: copy .env.example .env
        ```
    - Open `.env` and add your `OPENAI_API_KEY`.
    - (Optional) Add `TAVILY_API_KEY` if you want web search capabilities.

## Running the Agent

Run the main script:

```bash
python main.py
```

## Features

- Uses OpenAI's GPT models via LangChain.
- Includes a basic `multiply` tool as an example of custom tool usage.
- Supports conversational interaction.
