# End-to-End Document RAG System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Packaging](https://img.shields.io/badge/Packaging-pyproject.toml-green)

This repository implements an end-to-end Retrieval-Augmented Generation (RAG) system
for ingesting documents (PDFs, text files, and URLs) and querying them via an LLM-powered interface.

## Tech stack

- **Python** — core language
- **LangChain** — document loading, splitting, and orchestration
- **FAISS** — vector similarity search
- **OpenAI models** — embeddings and LLM inference
- **Streamlit** — interactive UI

## Steps to run the program:

### 1. Create and clone the repository

```bash
git clone https://github.com/Joab-David-Johanan/RAG_project_1
```

### 2. Make sure you have uv package manager

```bash
pip install uv # or: pipx install uv
```

### 3. Create a uv virtual environment

```bash
uv venv rag_env
```

```bash
rag_env\Scripts\activate
```

### 4. Install the project as a package in the uv virtual environment

```bash
uv pip install -e .
```

### 5. Run the streamlit app

```bash
rag-app
```

### 6. Roadmap

- Advanced chunking techniques
- Hybrid search strategies
- Query Enhancement techniques
- Multimodal RAG techniques
- Guardrails
- Cache for faster response times
- Deployment to AWS cloud with Docker
