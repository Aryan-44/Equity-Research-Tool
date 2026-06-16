# Equity Research Dashboard

This is a Streamlit-based web application that allows users to perform equity research by inputting news or article URLs. It uses LangChain, Google's Gemini LLMs, and FAISS for vector storage and retrieval-augmented generation (RAG).

## Features
- Input up to 3 URLs to extract text data.
- Processes text using unstructured loaders and splits it into chunks.
- Builds a FAISS vector database using Google Generative AI embeddings.
- Ask questions about the content of the provided URLs and get answers with sources.

## Setup

1. Clone the repository.
2. Create a virtual environment and install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory and add your Google API key:
   ```env
   google_api_key=YOUR_GOOGLE_API_KEY
   ```
4. Run the Streamlit application:
   ```bash
   streamlit run 1.py
   ```

## Note
The `.env` file containing your API key is ignored by git to keep it secure.
