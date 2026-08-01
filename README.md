# AI Agent Demo - Agentic AI and RAG Implementation

This project demonstrates **Agentic AI** and **RAG (Retrieval-Augmented Generation)** through two complementary Jupyter notebooks and supporting Python modules.

Both notebooks cover the same three concepts — basic LLM interaction, tool use, and document Q&A — but use different approaches:

| Notebook | Approach | Best for |
|----------|----------|----------|
| [`AI_Agent_test.ipynb`](AI_Agent_test.ipynb) | Native OpenRouter API + custom RAG classes | Learning HTTP/API fundamentals with minimal abstractions |
| [`Ai_Agent_with_Langchain.ipynb`](Ai_Agent_with_Langchain.ipynb) | LangChain framework (`ChatOpenRouter`, LCEL, agents, FAISS) | Learning LangChain patterns and building on its ecosystem |

---

## Jupyter Notebooks

### [`AI_Agent_test.ipynb`](AI_Agent_test.ipynb) — Native API Demo

This notebook implements everything with **direct API calls** and **custom Python modules**. No LangChain dependency is required for this notebook's core flow.

**Part 1: Basic AI Agent**
- Sends HTTP requests to the OpenRouter chat completions endpoint using `requests`
- Structures conversations with system and user messages
- Demonstrates model selection, temperature, and token limits
- Default model: `google/gemini-2.5-flash` (other free models listed in the notebook)

**Part 2: Function Calling (Tool Use)**
- Defines tools manually as OpenRouter-compatible JSON schemas
- Implements a two-step tool loop: LLM decides → function runs → LLM responds
- Includes four example tools: `calculate`, `get_current_time`, `text_uppercase`, `text_word_count`
- Includes error handling for models with limited function-calling support

**Part 3: RAG System**
- Uses [`VectorStore_v2.py`](VectorStore_v2.py) to process documents and build a JSON vector store
- Uses [`querykb_v2.py`](querykb_v2.py) for cosine similarity search and context-aware answers
- Supports multiple file formats (PDF, DOCX, Excel, TXT, JSON)
- Stores embeddings in `vectorstore/vector_store.json`

**Workflow:**
```
User Query → OpenRouter API (requests) → LLM Response
User Query → Tool schema → Execute Python function → Final response
User Question → Embed query → Cosine search → querykb_v2.RAG → Answer
```

---

### [`Ai_Agent_with_Langchain.ipynb`](Ai_Agent_with_Langchain.ipynb) — LangChain Tutorial (OpenRouter)

This notebook mirrors the same three-part structure using **LangChain abstractions**, based on the [GeeksforGeeks LangChain intro](https://www.geeksforgeeks.org/artificial-intelligence/introduction-to-langchain/), adapted to use OpenRouter instead of Google Gemini.

**Part 1: LangChain Basics**
- Uses `ChatOpenRouter` from `langchain-openrouter` (reads `OPENROUTER_API_KEY` from `.env`)
- Runs a simple prompt with `llm.invoke()`
- Builds a reusable **Prompt Template** and **LCEL chain**: `prompt_template | llm | StrOutputParser()`

**Part 2: LangChain Agent with Tools**
- Wraps the same four tools with the `@tool` decorator (schemas generated automatically)
- Uses `create_agent()` to handle the full tool-calling loop
- Includes the same demo queries as `AI_Agent_test.ipynb` plus an interactive cell

**Part 3: LangChain RAG**
- Loads PDFs from `docs/` with `PyPDFLoader`
- Splits text with `RecursiveCharacterTextSplitter` (500 tokens, 50 overlap)
- Creates embeddings via OpenRouter using `OpenAIEmbeddings`
- Stores vectors in a **FAISS** index at `vectorstore/langchain_faiss/`
- Answers questions with an LCEL RAG chain (retriever → prompt → LLM)

**Workflow:**
```
Prompt template → ChatOpenRouter → StrOutputParser
User query → create_agent() → @tool functions → Final answer
PDF docs → Split → Embed → FAISS → Retriever → RAG chain → Answer
```

---

### Which notebook should I use?

| Feature | `AI_Agent_test.ipynb` | `Ai_Agent_with_Langchain.ipynb` |
|---------|------------------------|-----------------------------------|
| LLM calls | Manual `requests.post` | `ChatOpenRouter.invoke()` |
| Chains | Not applicable | LCEL pipe syntax |
| Tool schemas | Hand-written JSON | Auto-generated from `@tool` |
| Tool loop | Manual two-step API calls | `create_agent()` |
| Vector store | Custom JSON + numpy | FAISS via LangChain |
| RAG pipeline | `VectorStore_v2.py` + `querykb_v2.py` | LangChain loaders + LCEL chain |
| Dependencies | Minimal (`requests`, `openai`, `numpy`) | LangChain ecosystem packages |
| File formats (RAG) | PDF, DOCX, Excel, TXT, JSON | PDF (via `PyPDFLoader`) |

**Recommendation:** Start with `AI_Agent_test.ipynb` to understand how APIs and RAG work under the hood, then move to `Ai_Agent_with_Langchain.ipynb` to see how LangChain simplifies the same patterns.

---

## Features

### Part 1: Basic AI Agent
- Direct API interaction with multiple LLM models via OpenRouter
- Support for various free models (Grok, GPT-OSS, DeepSeek, Gemma, Qwen)
- Configurable temperature and token limits
- Simple conversation interface

### Part 2: Function Calling (Tool Use)
- AI agents with external tool capabilities
- Example functions: mathematical calculations, time retrieval, text processing
- Automatic function selection by the AI model
- Error handling for models with limited function calling support

### Part 3: RAG System
- Document processing and semantic search
- Token-based text chunking with configurable overlap
- Vector embeddings via OpenRouter or OpenAI
- Context-aware question answering based on your documents

---

## Project Structure

```
AI_Agent_Demo_Codes/
├── AI_Agent_test.ipynb              # Native API demo (requests + custom RAG)
├── Ai_Agent_with_Langchain.ipynb    # LangChain tutorial (OpenRouter)
├── VectorStore_v2.py                # Native vector store creation and management
├── querykb_v2.py                    # Native RAG query interface
├── docs/                            # Documents folder for RAG knowledge base
├── vectorstore/
│   ├── vector_store.json            # Native JSON vector store (AI_Agent_test)
│   └── langchain_faiss/             # FAISS index (Ai_Agent_with_Langchain)
├── images/                          # Diagrams used in the LangChain notebook
├── requirements.txt                 # Python dependencies
├── env.example                      # Example environment variables
└── README.md                        # This file
```

---

## Prerequisites

- Python 3.7 or higher
- Jupyter Notebook (for running the demo notebooks)
- API Keys:
  - **OpenRouter API Key** (required for both notebooks)
    - Sign up at [OpenRouter.ai](https://openrouter.ai/)
    - Get your API key from the dashboard
  - **OpenAI API Key** (optional — only needed if using OpenAI directly instead of OpenRouter for embeddings)

---

## Installation

1. **Clone or download this repository**

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**

   Copy the example environment file and fill in your API keys:
   ```bash
   cp env.example .env
   ```

   Then edit the `.env` file and replace the placeholder values with your actual API keys:
   ```
   OPENROUTER_API_KEY=your_actual_openrouter_api_key
   OPENAI_API_KEY=your_actual_openai_api_key
   ```

   **Important**: Never commit your `.env` file to version control. It is already included in `.gitignore`.

---

## Usage

### Running the Notebooks

1. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Open either notebook:
   - **`AI_Agent_test.ipynb`** — native API approach
   - **`Ai_Agent_with_Langchain.ipynb`** — LangChain approach

3. Run cells sequentially from top to bottom

### Native notebook (`AI_Agent_test.ipynb`)

#### Part 1: Basic AI Agent

```python
MODEL = "google/gemini-2.5-flash"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is AI?"}
]
# Sent via requests.post to OpenRouter
```

**Available free models (examples):**
- `openai/gpt-oss-20b:free`
- `tngtech/deepseek-r1t2-chimera:free`
- `google/gemma-3-27b-it:free`
- `qwen/qwen3-coder:free`

#### Part 2: Function Calling

**Available functions:**
- `calculate(expression)`: Performs mathematical calculations
- `get_current_time()`: Returns current date and time
- `text_uppercase(text)`: Converts text to uppercase
- `text_word_count(text)`: Counts words in text

#### Part 3: RAG System

**Step 1 — Create vector store:**

```python
from VectorStore_v2 import VectorStore

store = VectorStore(
    api_provider='openrouter',
    embedding_model='openai/text-embedding-3-small',
    chunk_size=500,
    chunk_overlap=50
)
store.extract_save_vector_store("docs", "vectorstore/vector_store.json")
```

**Step 2 — Query the knowledge base:**

```python
from querykb_v2 import RAG

rag = RAG(
    vector_store_path="vectorstore/vector_store.json",
    api_provider='openrouter',
    embedding_model='openai/text-embedding-3-small',
    chat_model='google/gemini-2.5-flash'
)

answer, used_context = rag.askAI("Your question here", "You are a helpful assistant")
```

### LangChain notebook (`Ai_Agent_with_Langchain.ipynb`)

#### Part 1: LCEL chain

```python
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenRouter(model="google/gemini-2.5-flash", temperature=0.7)
prompt_template = PromptTemplate.from_template(
    "Give me 3 career skills that are in high demand in {year}."
)
chain = prompt_template | llm | StrOutputParser()
response = chain.invoke({"year": "2026"})
```

#### Part 2: Agent with tools

```python
from langchain_core.tools import tool
from langchain.agents import create_agent

agent = create_agent(model=llm, tools=[calculate, get_current_time, ...])
result = agent.invoke({"messages": [{"role": "user", "content": "What time is it?"}]})
```

#### Part 3: LangChain RAG

Build the FAISS index from PDFs in `docs/`, then query with the RAG LCEL chain. The index is saved to `vectorstore/langchain_faiss/` and can be reloaded on subsequent runs.

---

## Supported File Formats for RAG

| Format | Native (`VectorStore_v2.py`) | LangChain notebook |
|--------|------------------------------|--------------------|
| PDF (`.pdf`) | Yes | Yes |
| Word (`.docx`) | Yes | No (PDF only in notebook) |
| Excel (`.xlsx`, `.xls`) | Yes | No |
| Text (`.txt`) | Yes | No |
| JSON (`.json`) | Yes | No |

Place your documents in the `docs/` folder before building a vector store.

---

## Configuration

### Vector Store Parameters

- **chunk_size**: Maximum number of tokens per chunk (default: 1000 in code, 500 in notebooks)
- **chunk_overlap**: Number of tokens to overlap between chunks (default: 200 in code, 50 in notebooks)
- **batch_size**: Number of texts to process in each embedding batch (default: 16, native only)

### Model Parameters

- **temperature**: Controls randomness (0.2 = more focused, 1.0 = more creative)
- **max_tokens**: Maximum length of the response (default: 4096)

---

## How It Works

### Basic AI Agent
```
User Query → OpenRouter API → LLM Model → Response
```

### Function Calling
```
User Query → LLM Decides Function → Execute Function →
Return Result → LLM Generates Final Response
```

### RAG System
```
User Question → Embed Query → Similarity Search →
Retrieve Top-k Chunks → Combine with Question →
Send to LLM → Context-Aware Answer
```

---

## Key Differences: Basic AI vs RAG

| Feature | Basic AI Agent | RAG System |
|---------|---------------|------------|
| Knowledge Source | Pre-trained model | Your documents |
| Accuracy | General knowledge | Document-specific |
| Hallucinations | Possible | Reduced |
| Context Window | Limited | Can handle large docs |
| Use Case | General Q&A | Domain-specific Q&A |

---

## Troubleshooting

### Function Calling Errors
If you encounter errors with function calling:
- Some free models have limited or no function calling support
- Try switching to a different model (e.g., `google/gemini-2.5-flash` or `openai/gpt-oss-20b:free`)
- Check the error message for specific guidance

### Vector Store Creation Issues
- Ensure your OpenRouter (or OpenAI) API key is valid
- Check that documents in the `docs/` folder are readable
- Verify file formats are supported for the notebook you are using

### LangChain Import Errors
- Run `pip install -r requirements.txt` to install all LangChain packages
- The LangChain notebook also includes an install cell you can uncomment and run

### API Key Issues
- Ensure your `.env` file is in the project root
- Check that variable names match exactly: `OPENROUTER_API_KEY` and `OPENAI_API_KEY`
- Verify API keys are valid and not expired

---

## Author Information

**VectorStore_v2.py and querykb_v2.py:**
- Author: Usuf Com
- Contact: usufcom20@gmail.com
- Website: www.djamai.com
- LinkedIn: https://www.linkedin.com/in/usufcom

---

## License

Copyright (c) Clemios SARL

---

## Additional Notes

- **`AI_Agent_test.ipynb`** uses a native RAG implementation (`VectorStore_v2.py`, `querykb_v2.py`) without LangChain
- **`Ai_Agent_with_Langchain.ipynb`** uses LangChain's FAISS vector store and LCEL chains — a separate index from the native JSON store
- Both notebooks can use the same `docs/` folder, but each builds its own vector store format
- Embeddings default to `openai/text-embedding-3-small` via OpenRouter
- Chat completions default to `google/gemini-2.5-flash` via OpenRouter

---

## Next Steps

- Run both notebooks side by side and compare the native vs LangChain approaches
- Try different questions with the RAG systems
- Experiment with different chunk sizes and overlap values
- Adjust the `k` parameter (number of chunks retrieved)
- Try different free LLM models for comparison
- Add your own documents to the `docs/` folder
- Create custom functions for the function calling demos
