# Load environment variables from .env file
import os
import openai
from dotenv import load_dotenv

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

# Load environment variables from .env file
# load_dotenv()
load_dotenv(dotenv_path="/Users/scarbez-ai/Documents/Projects/_env/keys.env")

# Access the secrets safely
# api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")
# print( openai.api_key )

# Utility function for displaying data structures in a more readable format
def pretty_print(data):
    """Nicely prints any Python data structure."""
    pprint(data, indent=2, width=100, compact=False)


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


# Load the PDF and split into pages
# To-Do - Try PyPDF2 or other that might be better
# To-Do - Research and implement PDF data extractors and/or parsers that can use:
#   - the information in tables
#   - the information in diagrams
#   - textual image data
#   - non-textual image data

pdf_list = [
    "docs/AI-Driven-Business-Model-Innovation.pdf", 
    "docs/Business-Intelligence-Concepts-and-Approaches.pdf", 
    "docs/Time-Series-Data-Prediction-Using-IoT-and-ML.pdf", 
    "docs/Walmart_s-Sales-Data-Analysis.pdf"
    ]

pdf_data = []

for i, pdf in enumerate(pdf_list):
    loader = PyPDFLoader(pdf)
    pdf_data.append(loader.load())

# Apply document contemt type metadata for custom chunking
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


# Chunking
# Different chunks depending on document contenst type: scientific smaller chunks, more overlap; business, larger chunks, less overlap
# To-Do - Add auto-optimization capabilities

# To-Do - Add catch all / exceptions

pdf_chunks = []

for doc in pdf_data:
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
    
        pdf_chunks.extend(text_splitter.split_documents([doc_page]))


# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size= 500,
#     chunk_overlap=100 # 1/5 of the chunk size
# )

# split_doc = text_splitter.split_documents(pages)


final_chunks = []
buffer = ""
buffer_meta = None  # Optional: to keep first or combined metadata

for chunk in pdf_chunks:
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


# Define embedding model

embedding_model = OpenAIEmbeddings(model = "text-embedding-3-large") # Try larger model, 256 dimensions
# embedding_model = OpenAIEmbeddings(model = "text-embedding-3-small") # Original, used in demos, 128 dimensions


# Build vector store from split_doc using your embedding model
vector_store = Chroma.from_documents(final_chunks, embedding_model)

# To-Do - Move initialization to the best place
k_context_chunks = 4

# Create a retriever with desired parameters
retriever = vector_store.as_retriever(search_kwargs={"k": k_context_chunks})