# ConversationalRetrievalChain chat to use with the RAG agent tool to have better responses when researching
#   BI concepts and practices within the PDF documents

# Load environment variables from .env file
import os
import openai
from dotenv import load_dotenv

from langchain.chains import ConversationalRetrievalChain, LLMChain #, load_qa_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from agent.rag import get_retriever # This returns a retriever for the RAG tool to use

# Load environment variables from .env file
load_dotenv(dotenv_path="/Users/scarbez-ai/Documents/Projects/_env/keys.env")

# Set the OpenAI key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)

# Return k value for context chunks
def get_k_context_chunks():
    return k_context_chunks

# Retriever
retriever = get_retriever()

# Set up memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    output_key="answer",
    return_messages=True 
)

# Prompt for rephrasing (question condensing)
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""
    Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
    Chat History: {chat_history}
    Follow Up Input: {question}
    Standalone question:
""")

question_generator = LLMChain(
    llm=llm,
    prompt=CONDENSE_QUESTION_PROMPT
)

# Chain for answering using retrieved docs
combine_docs_chain = load_qa_chain(
    llm,
    chain_type="stuff"
)

# Build the ConversationalRetrievalChain
qa_chain = ConversationalRetrievalChain(
    retriever=retriever,
    memory=memory,
    question_generator=question_generator,
    combine_docs_chain=combine_docs_chain,
    return_source_documents=True,
    verbose=True
)

# Chat loop for testing only. In reality the agent would run the chain, not the chat loop.
def run_rag_chat():
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break
        response = qa_chain.invoke({"question": user_input})
        print("Assistant:", response["answer"])