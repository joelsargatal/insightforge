# Load environment variables from .env file
import os
import openai
from dotenv import load_dotenv

# Data structures and metadata
import json
from pathlib import Path
from datetime import datetime

# Display formatting
from pprint import pprint

# PDF loading
from langchain.document_loaders import PyPDFLoader

# Text splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# OpenAI embeddings
from langchain_openai.embeddings import OpenAIEmbeddings

# Vector store
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma

# Plotting
import matplotlib.pyplot as plt

# Load environment variables from .env file
load_dotenv(dotenv_path="/Users/scarbez-ai/Documents/Projects/_env/keys.env")

# Set the OpenAI key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Location for documents and metadata file
DOCS_DIR = Path("docs/")
META_FILE = DOCS_DIR / "metadata.json"

# k value for context chunks
k_context_chunks = 4

# Utility function for displaying data structures in a more readable format
def pretty_print(data):
    """Nicely prints any Python data structure."""
    pprint(data, indent=2, width=100, compact=False)

# Return k value for context chunks. Used by the RAG tool
def get_k_context_chunks():
    return k_context_chunks

# Function to get return a retriever to the RAG chat tool
def get_retriever(k: int = k_context_chunks):
    vector_store = load_or_rebuild_vector_store()
    return vector_store.as_retriever(search_kwargs={"k": k})

# Auxiliary function to check chunk sizes and distribution to see if merging is needed
#   and the result of chunking optimization
# This is only for debugging purposes and should not be called in production

def disp_chunk_distr(chunk_list):
    # Get chunk sizes (number of characters)
    chunk_sizes = [len(chunk.page_content) for chunk in chunk_list]

    # Basic statistics
    print(f"Number of chunks: {len(chunk_sizes)}")
    print(f"Mean chunk size: {sum(chunk_sizes)/len(chunk_sizes):.2f}")
    print(f"Min chunk size: {min(chunk_sizes)}")
    print(f"Max chunk size: {max(chunk_sizes)}")

    # Plot histogram
    plt.figure(figsize=(10, 5))
    plt.hist(chunk_sizes, bins=30, color='skyblue', edgecolor='black')
    plt.title("Distribution of Chunk Sizes")
    plt.xlabel("Chunk Size (characters)")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.75)
    plt.show()

# Optimize for best use of context window for sending chunked context to LLM
def build_context_from_chunks(results, max_chunks):
    """
    Takes the similarity search results with scores, sorts them, and returns a context string
    for the LLM prompt.
    """
    # Sort by descending similarity score
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

    # Keep only the top N
    top_chunks = sorted_results[:max_chunks]

    # Merge their contents
    context = "\n\n".join(doc.page_content for doc, score in top_chunks)

    return context

# Dynamically load the PDF documents
# To-Do - Try PyPDF2 or other that might be better
# To-Do - Research and implement PDF data extractors and/or parsers that can use:
#   - the information in tables
#   - the information in diagrams
#   - textual image data
#   - non-textual image data

def load_docs(docs_paths):
    # Original PDF document list for reference
    # pdf_list = [
    #     "Data/BI PDFs/AI-Driven-Business-Model-Innovation.pdf", 
    #     "Data/BI PDFs/Business-Intelligence-Concepts-and-Approaches.pdf", 
    #     "Data/BI PDFs/Time-Series-Data-Prediction-Using-IoT-and-ML.pdf", 
    #     "Data/BI PDFs/Walmart_s-Sales-Data-Analysis.pdf"
    #     ]

    docs_data = []

    for i, path in enumerate(docs_paths):
        loader = PyPDFLoader(str(path))
        docs_data.append(loader.load())

    return docs_data

# Apply document content type metadata for custom chunking
def classify_docs(pdf_data):
    for doc in range(len(pdf_data)):
        for page in pdf_data[doc]:
            # To-Do - Add catch all / exceptions
            match doc:
                case 0:
                    page.metadata["doc_type"] = "academic"
                case 1:
                    page.metadata["doc_type"] = "academic"
                case 2:
                    page.metadata["doc_type"] = "academic"
                case 3:
                    page.metadata["doc_type"] = "business_case"

    return pdf_data

# Chunking
# Different chunks depending on document contenst type: scientific smaller chunks, more overlap; business, larger chunks, less overlap
# To-Do - Add auto-optimization capabilities
# To-Do - Add catch all / exceptions

def chunk_docs(docs_data):
    docs_chunks = []

    for doc in docs_data:
        for doc_page in doc:
            doc_type = doc_page.metadata.get("doc_type", "business")

            match doc_type:
                case "scientific":
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size= 500,
                        chunk_overlap=100 # 1/5 of the chunk size
                )
                case "business":
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size= 1000,
                        chunk_overlap=200 # 1/5 of the chunk size
                )
                case _:
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=800,
                        chunk_overlap=160
                    )
        
            docs_chunks.extend(text_splitter.split_documents([doc_page]))

    return docs_chunks

# Optimize chunks sizings
def optimize_chunks(docs_chunks):
    final_chunks = []
    buffer = ""
    buffer_meta = None  # Optional: to keep first or combined metadata

    for chunk in docs_chunks:
        text = chunk.page_content.strip()
        if len(text) < 300:
            if not buffer:
                buffer = text
                buffer_meta = chunk.metadata  # start new buffer
            else:
                buffer += " " + text
                # Optionally, update metadata to combine source info here
            # Only create chunk when buffer is big enough
            if len(buffer) >= 500:
                final_chunks.append(Document(page_content=buffer.strip(), metadata=buffer_meta))
                buffer = ""
                buffer_meta = None
        else:
            # If there is something in the buffer, flush it before appending the large chunk
            if buffer:
                final_chunks.append(Document(page_content=buffer.strip(), metadata=buffer_meta))
                buffer = ""
                buffer_meta = None
            final_chunks.append(chunk)

    # If any text remains in the buffer, flush it as a final chunk
    if buffer:
        final_chunks.append(Document(page_content=buffer.strip(), metadata=buffer_meta))

    return final_chunks

# Functions for building and rebuilding the vector stores if documents have changed
def get_docs_paths():
    return [f for f in DOCS_DIR.glob("*.pdf")]

def get_file_info(file_path):
    stat = os.stat(file_path)
    return {
        "size": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
    }

def needs_rebuild(docs_paths):
    if not META_FILE.exists():
        return True

    with open(META_FILE, "r") as f:
        old_meta = json.load(f)

    for file in docs_paths:
        info = get_file_info(file)
        name = file.name

        if name not in old_meta or old_meta[name] != info:
            return True

    return False

def save_metadata(docs_paths):
    current_meta = {
        f.name: get_file_info(f)
        for f in docs_paths
    }
    with open(META_FILE, "w") as f:
        json.dump(current_meta, f, indent=2)

def load_or_rebuild_vector_store():
    # Define embedding model
    embedding_model = OpenAIEmbeddings(model = "text-embedding-3-large") # Try larger model, 256 dimensions
    # embedding_model = OpenAIEmbeddings(model = "text-embedding-3-small") # Original, used in demos, 128 dimensions

    docs_paths = get_docs_paths()
    docs_chunks = []

    # To-Do - Maybe nest calls or something
    if needs_rebuild(docs_paths):
        print("🔁 Rebuilding vector store...")
        docs_data = load_docs(docs_paths)
        docs_data = classify_docs(docs_data)
        docs_chunks = chunk_docs(docs_data)
        docs_chunks = optimize_chunks(docs_chunks)
        save_metadata(docs_paths)
        vector_store = Chroma.from_documents(
            docs_chunks,
            embedding_model,
            persist_directory="db" # Needed for ConversationalRetrievalChain routing to work
        )
        vector_store.persist()
    else:
        print("✅ Loading existing vector store...")
        # Build vector store from split_doc using your embedding model
        vector_store = Chroma(persist_directory="db", embedding_function=embedding_model)

    return vector_store

# Run the load or rebuild function once, at startup, so when the application flow is routed to
#   the RAG tool, there is no document loading delay.
# We keep the code in a function instead of having it as main, so we can choose to run the
#   load_or_rebuild_vector_store() function again at different points, like every time we route
#   to the RAG tool, to be able to use newly uploaded documents during runtime. We are not doing
#   this, for this demo though.

vector_store = load_or_rebuild_vector_store()