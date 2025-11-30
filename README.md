# AI Agent Demo - Agentic AI and RAG Implementation

This project demonstrates two key AI concepts through practical implementations:

1. **Agentic AI**: Direct interaction with Large Language Models (LLMs) via OpenRouter API
2. **RAG (Retrieval-Augmented Generation)**: Enhanced AI responses using document knowledge bases

## Features

### Part 1: Basic AI Agent
- Direct API interaction with multiple free LLM models via OpenRouter
- Support for various free models (Grok, GPT-OSS, DeepSeek, Gemma, Qwen)
- Configurable temperature and token limits
- Simple conversation interface

### Part 2: Function Calling (Tool Use)
- AI agents with external tool capabilities
- Example functions: mathematical calculations, time retrieval, text processing
- Automatic function selection by the AI model
- Error handling for models with limited function calling support

### Part 3: RAG System
- Document processing from multiple formats (PDF, DOCX, Excel, TXT, JSON)
- Token-based text chunking with configurable overlap
- Vector embeddings using OpenAI's embedding models
- Cosine similarity search for document retrieval
- Context-aware question answering based on your documents

## Project Structure

```
AI_Agent_Demo_Codes/
├── AI_Agent_test.ipynb          # Main Jupyter notebook with all demos
├── VectorStore_v2.py            # Vector store creation and management
├── querykb_v2.py                 # RAG query interface
├── docs/                         # Documents folder for RAG knowledge base
│   └── ETIIAC_2025_forRAG.pdf
├── vectorstore/                  # Generated vector store storage
│   └── vector_store.json
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Prerequisites

- Python 3.7 or higher
- Jupyter Notebook (for running the demo notebook)
- API Keys:
  - **OpenRouter API Key**: For accessing free LLM models
    - Sign up at [OpenRouter.ai](https://openrouter.ai/)
    - Get your API key from the dashboard
  - **OpenAI API Key**: For embeddings and RAG system
    - Sign up at [OpenAI](https://platform.openai.com/)
    - Get your API key from the API keys section

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

## Usage

### Running the Notebook

1. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Open `AI_Agent_test.ipynb`

3. Run cells sequentially to see each demo in action

### Part 1: Basic AI Agent

The basic AI agent demonstrates simple conversation with LLMs:

```python
# Example from the notebook
MODEL = "x-ai/grok-4.1-fast:free"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me something about AI in Africa."}
]
```

**Available Free Models:**
- `x-ai/grok-4.1-fast:free`
- `openai/gpt-oss-20b:free`
- `tngtech/deepseek-r1t2-chimera:free`
- `google/gemma-3-27b-it:free`
- `qwen/qwen3-coder:free`

### Part 2: Function Calling

The function calling demo shows how AI agents can use external tools:

**Available Functions:**
- `calculate(expression)`: Performs mathematical calculations
- `get_current_time()`: Returns current date and time
- `text_uppercase(text)`: Converts text to uppercase
- `text_word_count(text)`: Counts words in text

The AI automatically decides when to use these functions based on the user's query.

### Part 3: RAG System

#### Step 1: Create Vector Store

Before querying documents, you need to create a vector store:

```python
from VectorStore_v2 import VectorStore
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Vector Store
store = VectorStore(
    api_key=os.getenv("OPENAI_API_KEY"),
    chunk_size=500,
    chunk_overlap=50
)

# Create and save vector store
kb_folder = "docs"
vector_store_path = "vectorstore/vector_store.json"
store.exract_save_vector_store(store, kb_folder, vector_store_path)
```

#### Step 2: Query the Knowledge Base

Once the vector store is created, you can query it:

```python
from querykb_v2 import RAG
import textwrap

# Initialize RAG system
rag = RAG(vector_store_path="vectorstore/vector_store.json")

# Ask questions
system_prompt = "You are a helpful assistant"
user_msg = "What is this document about?"

answer, used_context = rag.askAI(user_msg, system_prompt, k=3)

print(f"Reply: {textwrap.fill(answer, width=80)}")
```

**Parameters:**
- `k`: Number of document chunks to retrieve (default: 5, shown example uses 3)
- `chunk_size`: Maximum tokens per chunk (default: 500)
- `chunk_overlap`: Token overlap between chunks (default: 50)

## Supported File Formats for RAG

The RAG system supports the following document formats:

- **PDF** (`.pdf`): Extracts text from each page with page numbers
- **Word Documents** (`.docx`): Extracts paragraphs and tables separately
- **Excel Files** (`.xlsx`, `.xls`): Converts each sheet to text with sheet information
- **Text Files** (`.txt`): Direct text extraction
- **JSON Files** (`.json`): Converts to string representation

Place your documents in the `docs/` folder before creating the vector store.

## Configuration

### Vector Store Parameters

- **chunk_size**: Maximum number of tokens per chunk (default: 1000 in code, 500 in notebook)
- **chunk_overlap**: Number of tokens to overlap between chunks (default: 200 in code, 50 in notebook)
- **batch_size**: Number of texts to process in each embedding batch (default: 16)

### Model Parameters

- **temperature**: Controls randomness (0.2 = more focused, 1.0 = more creative)
- **max_tokens**: Maximum length of the response (default: 4096)

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

## Key Differences: Basic AI vs RAG

| Feature | Basic AI Agent | RAG System |
|---------|---------------|------------|
| Knowledge Source | Pre-trained model | Your documents |
| Accuracy | General knowledge | Document-specific |
| Hallucinations | Possible | Reduced |
| Context Window | Limited | Can handle large docs |
| Use Case | General Q&A | Domain-specific Q&A |

## Troubleshooting

### Function Calling Errors
If you encounter errors with function calling:
- Some free models have limited or no function calling support
- Try switching to a different model (e.g., `openai/gpt-oss-20b:free`)
- Check the error message for specific guidance

### Vector Store Creation Issues
- Ensure your OpenAI API key is valid and has sufficient credits
- Check that documents in the `docs/` folder are readable
- Verify file formats are supported

### API Key Issues
- Ensure your `.env` file is in the project root
- Check that variable names match exactly: `OPENROUTER_API_KEY` and `OPENAI_API_KEY`
- Verify API keys are valid and not expired

## Author Information

**VectorStore_v2.py and querykb_v2.py:**
- Author: Usuf Com
- Contact: usufcom20@gmail.com
- Website: www.djamai.com
- LinkedIn: https://www.linkedin.com/in/usufcom

## License

Copyright (c) Clemios SARL

## Additional Notes

- The RAG implementation is a native solution without external dependencies like Langchain
- Vector stores are saved as JSON files for easy inspection and portability
- The system uses cosine similarity for document retrieval
- Embeddings are created using OpenAI's `text-embedding-3-small` model
- Chat completions use `gpt-4o-mini` for optimal performance and cost

## Next Steps

- Try different questions with the RAG system
- Experiment with different chunk sizes and overlap values
- Adjust the `k` parameter (number of chunks retrieved)
- Try different free LLM models for comparison
- Add your own documents to the `docs/` folder
- Create custom functions for the function calling demo

