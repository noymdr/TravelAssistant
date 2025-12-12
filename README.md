# Travel Assistant RAG System

A Retrieval-Augmented Generation (RAG) system for querying travel documents (flight tickets, insurance details, hotel information) using local models and vector storage.

## Features

- **Source-First Retrieval**: Prioritizes information from your provided documents
- **Intelligent Enrichment**: Automatically enriches answers with LLM knowledge when information isn't in your documents
- **Fully Local**: Uses Hugging Face embeddings and Ollama (Gemma 2) for complete privacy
- **FAISS Vector Store**: Fast and efficient local vector storage
- **Multiple Document Types**: Supports PDF and text files

## Requirements

- Python 3.8+
- Ollama installed and running with Gemma 2 model
- Internet connection (for first-time model downloads)

## Installation

1. **Create and activate a virtual environment (recommended):**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # On Windows (Command Prompt):
   .\venv\Scripts\activate.bat
   # On macOS/Linux:
   source venv/bin/activate
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install and setup Ollama:**
   - Download Ollama from https://ollama.ai
   - Install and start Ollama
   - Pull the Gemma 2 model:
     ```bash
     ollama pull gemma2
     ```

4. **Prepare your documents:**
   - Create a `documents` folder in the project root
   - Add your travel documents (PDF or TXT files):
     - Flight ticket (e.g., `flight_ticket.pdf`)
     - Insurance details (e.g., `insurance.pdf`)
     - Hotel information (e.g., `hotel.txt`)

## Usage

### Basic Usage

```python
from rag_system import TravelRAGSystem

# Initialize the system
rag = TravelRAGSystem(
    documents_dir="documents",
    vector_store_path="vector_store",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    ollama_base_url="http://localhost:11434",
    llm_model="gemma2"
)

# Query the system
result = rag.query("When is my flight?")
print(result['answer'])
```

### Interactive Mode

Run the example script for an interactive query interface:

```bash
python example_usage.py
```

### Example Queries

1. **Source-based queries** (answered from your documents):
   - "When is my flight?"
   - "What's the hotel phone number?"
   - "What is my insurance policy number?"

2. **Enriched queries** (uses documents + LLM knowledge):
   - "What are the reviews about the hotel?"
   - "What's the weather like at my destination?"
   - "What are some popular restaurants near my hotel?"

## How It Works

1. **Document Loading**: The system loads all PDF and text files from the `documents` directory
2. **Chunking**: Documents are split into smaller chunks for better retrieval
3. **Embedding**: Each chunk is converted to a vector using Hugging Face embeddings
4. **Vector Store**: FAISS stores all document embeddings for fast similarity search
5. **Query Processing**:
   - Retrieves relevant document chunks based on query similarity
   - Checks if the answer can be found in the retrieved documents
   - If yes: Answers from documents only
   - If no: Enriches with LLM knowledge while using document context

## Project Structure

```
TravelAssistant/
├── documents/              # Place your travel documents here
│   ├── flight_ticket.pdf
│   ├── insurance.pdf
│   └── hotel.txt
├── vector_store/          # FAISS index and document metadata (auto-generated)
├── rag_system.py          # Main RAG system implementation
├── example_usage.py       # Example usage script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Configuration

You can customize the system by modifying the initialization parameters:

- `documents_dir`: Directory containing your documents
- `vector_store_path`: Where to save/load the vector store
- `embedding_model`: Hugging Face embedding model (default: `all-MiniLM-L6-v2`)
- `ollama_base_url`: Ollama API URL (default: `http://localhost:11434`)
- `llm_model`: Ollama model name (default: `llama3`)

## Rebuilding the Index

If you add new documents, rebuild the index:

```python
rag.rebuild_index()
```

## Troubleshooting

1. **Ollama connection error**: Make sure Ollama is running:
   ```bash
   ollama serve
   ```

2. **Model not found**: Pull the required model:
   ```bash
   ollama pull llama3
   ```

3. **No documents found**: Ensure your documents are in the `documents` folder and are PDF or TXT files

4. **Slow first run**: The first run downloads the embedding model (~80MB), subsequent runs are faster

## License

This project is provided as-is for personal use.


