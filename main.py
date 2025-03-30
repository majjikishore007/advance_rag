import os
import logging
import time
from typing import List
from urllib.request import urlretrieve

from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from langchain.docstore.document import Document as LangchainDocument
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone.grpc import PineconeGRPC as Pinecone  
from pinecone import ServerlessSpec
from transformers import AutoTokenizer
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("rag_service.log")]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 256
BATCH_SIZE = 100  # Adjust based on needs
MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]


class RagService:
    """A service class for processing documents and managing a RAG pipeline with Pinecone."""

    def __init__(self, pinecone_api_key: str, index_name: str = "rag-index"):
        logger.info("Initializing RagService with Pinecone gRPC and embedding model")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_ID,
            multi_process=True,
            encode_kwargs={"normalize_embeddings": True},
        )
        self.tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
        self.pinecone_client = Pinecone(api_key=pinecone_api_key)
        self.index_name = index_name
        self._initialize_pinecone_index()

    def _initialize_pinecone_index(self):
        logger.info(f"Checking if Pinecone index '{self.index_name}' exists")
        if self.index_name not in self.pinecone_client.list_indexes().names():
            logger.info(f"Creating new Pinecone index: {self.index_name}")
            self.pinecone_client.create_index(
                name=self.index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            logger.info(f"Successfully created Pinecone index: {self.index_name}")
        self.index = self.pinecone_client.Index(self.index_name)
        logger.debug(f"Connected to Pinecone index: {self.index_name}")

    def _download_document(self, url: str) -> str:
        logger.info(f"Downloading document from URL: {url}")
        start_time = time.time()
        local_filename = os.path.basename(url)
        try:
            urlretrieve(url, local_filename)
            logger.info(f"Downloaded document to: {local_filename} in {time.time() - start_time:.2f}s")
            return local_filename
        except Exception as e:
            logger.error(f"Failed to download document from {url}: {e}")
            raise

    def _parse_document(self, file_path: str) -> List[LangchainDocument]:
        logger.info(f"Parsing document: {file_path}")
        start_time = time.time()
        try:
            converter = DocumentConverter()
            result = converter.convert(file_path)
            markdown_content = result.document.export_to_markdown()
            doc = [LangchainDocument(page_content=markdown_content, metadata={"source": file_path})]
            logger.info(f"Parsed document in {time.time() - start_time:.2f}s")
            return doc
        except Exception as e:
            logger.error(f"Failed to parse document {file_path}: {e}")
            raise ValueError(f"Failed to parse document: {e}")

    def _split_documents(self, documents: List[LangchainDocument], chunk_size: int) -> List[LangchainDocument]:
        logger.info(f"Splitting {len(documents)} documents into chunks")
        start_time = time.time()
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            self.tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size / 10),
            add_start_index=True,
            strip_whitespace=True,
            separators=MARKDOWN_SEPARATORS,
        )
        docs_processed = []
        for doc in documents:
            docs_processed += text_splitter.split_documents([doc])

        unique_texts = {}
        docs_processed_unique = [doc for doc in docs_processed if not (doc.page_content in unique_texts or unique_texts.update({doc.page_content: True}))]
        logger.info(f"Processed {len(docs_processed_unique)} unique chunks in {time.time() - start_time:.2f}s")
        return docs_processed_unique

    def _embed_documents(self, documents: List[LangchainDocument]) -> List[dict]:
        """Embed documents in batches and prepare vectors for upsert."""
        logger.info(f"Embedding {len(documents)} documents")
        start_time = time.time()
        vectors = []
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding_model.embed_documents(texts)  # Batch embedding
        for i, (embedding, doc) in enumerate(zip(embeddings, documents)):
            vectors.append({
                "id": f"doc_{i}",
                "values": embedding,
                "metadata": {
                    "source": doc.metadata["source"],
                    "text": doc.page_content  # Include text in metadata
                }
            })
        logger.info(f"Embedded {len(vectors)} documents in {time.time() - start_time:.2f}s")
        return vectors

    def _upsert_vectors(self, vectors: List[dict]):
        """Upsert vectors in batches using Pinecone gRPC."""
        logger.info(f"Upserting {len(vectors)} vectors into Pinecone using gRPC")
        start_time = time.time()

        # Split vectors into batches
        batches = [vectors[i:i + BATCH_SIZE] for i in range(0, len(vectors), BATCH_SIZE)]
        for batch in tqdm(batches, desc="Upserting batches"):
            try:
                self.index.upsert(vectors=batch)
                logger.debug(f"Upserted batch of {len(batch)} vectors")
            except Exception as e:
                logger.error(f"Failed to upsert batch: {e}")
                raise

        logger.info(f"Upserted all vectors in {time.time() - start_time:.2f}s")

    def process_document(self, document_url: str):
        logger.info(f"Starting document processing for URL: {document_url}")
        try:
            # file_path = self._download_document(document_url)
            raw_docs = self._parse_document(document_url)
            docs_processed = self._split_documents(raw_docs, MAX_TOKENS)
            vectors = self._embed_documents(docs_processed)
            self._upsert_vectors(vectors)
            logger.info(f"Successfully processed and stored document from {document_url}")
        except Exception as e:
            logger.error(f"Error processing document from {document_url}: {e}")
            raise

    def query(self, user_query: str, k: int = 5, model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1") -> str:
        logger.info(f"Processing query: '{user_query}'")
        try:
            start_time = time.time()
            query_vector = self.embedding_model.embed_query(user_query)
            results = self.index.query(vector=query_vector, top_k=k, include_metadata=True)
            retrieved_docs = [match.metadata["text"] for match in results.matches]  # Access text from metadata
            # print(retrieved_docs)
            logger.info(f"Retrieved {len(retrieved_docs)} documents from Pinecone")
            logger.debug(f"Top retrieved document: {retrieved_docs[0][:200]}...")
            context = "\n".join(retrieved_docs)
            prompt = f"Given the following context:\n{context}\n\nAnswer the query: {user_query} note: please ansewer if the query is related to the context else respond with I dont know"
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0,
                max_tokens=MAX_TOKENS,
                timeout=None,
                max_retries=2,
            )
            response = llm.invoke(prompt)
            # response = completion(model=model, messages=[{"role": "user", "content": prompt}], max_tokens=MAX_TOKENS)
            # print("response ---",response)
            answer = response.content
            logger.info(f"Query processed in {time.time() - start_time:.2f}s")
            return answer
        except Exception as e:
            logger.error(f"Error querying the knowledge base: {e}")
            return f"Error querying the knowledge base: {e}"


if __name__ == "__main__":
    try:

        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            logger.error("PINECONE_API_KEY not found in environment variables")
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        rag_service = RagService(pinecone_api_key=pinecone_api_key,index_name="rag-index")
        document_url = "data/dubai.pdf"  # Replace with a real URL
        # rag_service.process_document(document_url)
        user_query = [
            "Can you list down the limits for microbial contamination and toxic heavy metals for personal care products?",
            "what is the toxic heavy metals for skin care in india ?"
        ]
        # user_query = "Can you list down the limits for microbial contamination and toxic heavy metals for personal care products?"
        for qu in user_query:
            response = rag_service.query(qu,model="gemini-2.0-flash")
            print(f"Response: {response}")
            print()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        print(f"Error: {e}")