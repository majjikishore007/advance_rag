# Advance RAG Service

This project implements a Retrieval-Augmented Generation (RAG) pipeline using Pinecone for vector storage, HuggingFace for embeddings, and Google Generative AI for query answering. It processes documents, stores their embeddings in Pinecone, and allows querying the knowledge base with natural language.

## Features

- **Document Processing**: Downloads, parses, and splits documents into smaller chunks for efficient embedding.
- **Embeddings**: Uses HuggingFace's `sentence-transformers/all-MiniLM-L6-v2` model for generating embeddings.
- **Vector Storage**: Stores embeddings in Pinecone for efficient similarity search.
- **Query Answering**: Retrieves relevant documents and generates answers using Google Generative AI.
- **Logging**: Comprehensive logging for debugging and monitoring.

## Project Structure

```
.env                # Environment variables
main.py             # Main script containing the RAG service implementation
Readme.md           # Project documentation
requirements.txt    # Python dependencies
```

## Prerequisites

- Python 3.8 or higher
- A Pinecone account and API key
- HuggingFace Transformers library
- Google Generative AI access

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd advance_rag
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the `.env` file:
   ```plaintext
   PINECONE_API_KEY=<your-pinecone-api-key>
   GOOGLE_API_KEY=<your-google-api-key>
   ```

## Usage

### Document Processing

To process a document and store its embeddings in Pinecone:
1. Update the `document_url` in `main.py` with the path or URL of the document.
2. Run the script:
   ```bash
   python main.py
   ```

### Querying

To query the knowledge base:
1. Update the `user_query` list in `main.py` with your questions.
2. Run the script:
   ```bash
   python main.py
   ```

### Example Query

```plaintext
Query: What are the benefits of using RAG pipelines?
Response: <Generated response from the AI model>
```

## Environment Variables

The following environment variables are required:

- `PINECONE_API_KEY`: Your Pinecone API key.
- `GOOGLE_API_KEY`: Your Google Generative AI API key.

## Dependencies

The project uses the following libraries:

- `langchain`
- `pinecone`
- `transformers`
- `tqdm`
- `python-dotenv`
- `logging`

## Logging

Logs are stored in `rag_service.log` and also printed to the console. Adjust the logging level in `main.py` as needed.

## Future Enhancements

- Add support for additional document formats.
- Improve error handling and retries.
- Optimize embedding and upsert batch sizes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.