# Streamlit Application for Retrieval-Augmented Generation (RAG)

This repository contains a Streamlit application for performing Retrieval-Augmented Generation (RAG) tasks using LlamaIndex, OpenAI models, and various utility functions. The application allows users to configure parameters, retrieve relevant document chunks, and generate responses using advanced language models.

## Features
- **Environment Configuration:** Load environment variables and configure API keys.
- **Chunked Document Retrieval:** Retrieve relevant document chunks based on user queries.
- **Answer Generation:** Generate answers using OpenAI's language models based on the retrieved document chunks.
- **Customizable Parameters:** Configure chunk size, chunk overlap, number of chunks retrieved, and prompt templates.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/streamlit-rag-app.git
   cd streamlit-rag-app
   OPENAI_API_KEY=your_openai_api_key
   ```
## Usage

1. **Run the Streamlit application:**
   ```bash
   streamlit run st_app.py
   ```

2. **Configure Parameters:**

- **Select a Book:** Choose a book from the dropdown menu.
- **Input Query:** Enter your query in the provided text box.
- **Model Parameters:** Select the OpenAI model, chunk size, chunk overlap, and number of chunks to retrieve.
- **Prompt Template:** Edit the prompt template as needed.
