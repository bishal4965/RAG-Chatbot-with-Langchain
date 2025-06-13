# RAG Chatbot using Langchain with Appointment Booking Feature

This is a Streamlit-based chatbot that can:

* Answer questions from uploaded documents using LangChain and ChromaDB
* Collect user information to book appointments via conversation
* Handle general conversation using a Groq-hosted LLM (Mixtral)

## Features

* Document-based question answering (QA)
* Conversational form for appointment booking
* ReAct agent that decides whether to use a tool or respond directly
* Embedding-based document retrieval using HuggingFace + Chroma

## Tech Stack

* Streamlit for the web interface
* LangChain for agents, tools, and memory
* HuggingFace for embeddings
* ChromaDB for storing vectorized document chunks
* Groq LLM (Mixtral) for responses


## Setup Instructions

1. Clone the repository

```bash
git clone https://github.com/bishal4965/RAG-Chatbot-with-Langchain.git
cd RAG-Chatbot-with-Langchain
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Add environment variables in a `.env` file:

```
GROQ_API_KEY=your_groq_api_key
CHROMA_DB_PATH=./data/chroma_db
LOG_LEVEL=DEBUG
```

4. Run the app

```bash
streamlit run main.py
```

## How It Works

### Document QA

* Splits uploaded documents into chunks
* Computes embeddings using HuggingFace
* Stores them in ChromaDB
* Retrieves top relevant chunks to answer user questions

### Appointment Booking

* Initiated by messages like "book appointment"
* Collects name, email, phone number, and date
* Returns a summary once details are collected

### Agent Behavior

* Uses LangChain's ReAct agent to choose between:

  * Document QA tool
  * Appointment booking tool
  * Direct LLM response


## License

MIT License
