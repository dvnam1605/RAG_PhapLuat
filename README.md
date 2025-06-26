# Vietnamese Legal Document RAG System

This project implements a Retrieval-Augmented Generation (RAG) system for Vietnamese legal documents. The system uses local embeddings, vector stores, and query transformation techniques to provide accurate answers to legal questions based on a corpus of Vietnamese legal texts.

## Overview

The system retrieves relevant legal document passages based on user queries and uses a Large Language Model (Gemini) to generate accurate responses grounded in the retrieved context. It features multiple query transformation methods to improve retrieval quality:

- **Rewrite**: Makes queries more specific with legal terminology
- **Step Back**: Generalizes queries to capture broader legal concepts
- **Decompose**: Breaks complex queries into simpler sub-queries

## Components

- **Vector Store**: FAISS for efficient similarity search
- **Embeddings**: GPT4All for local embedding generation
- **LLM**: Google Gemini API for response generation
- **Interface**: Streamlit web interface for easy interaction
- **Data**: Collection of Vietnamese legal documents

## Project Structure

```
├── app.py              # Streamlit web interface
├── qabot.py            # Main RAG implementation
├── query_transform.py  # Query transformation methods
├── model/              # Embedding models
├── vector_store/       # FAISS vector store files
├── output_texts/       # Processed legal documents
└── crawler/            # Data collection utilities
```

## Setup and Installation

### Prerequisites

- Python 3.9+
- Git
- [Optional] CUDA-compatible GPU for faster embedding

### Installation

1. Clone this repository:

   ```
   https://github.com/dvnam1605/RAG_PhapLuat.git
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your Google API key:

   ```
   API_KEY=your_google_api_key_here
   ```

4. Run the Streamlit interface:
   ```
   streamlit run app.py
   ```

## Usage

1. Enter a legal question in Vietnamese in the text box
2. Select a query transformation method (optional)
3. Click "Gửi" to get an answer
4. Explore retrieved legal contexts in the expandable sections

## Query Transformation Methods

- **Rewrite**: Enhances queries with specific legal terminology and concepts
- **Step Back**: Creates a more general version of the query to capture broader legal principles
- **Decompose**: Breaks complex legal queries into focused sub-questions


## Acknowledgements

- [Add credits and acknowledgements here]
