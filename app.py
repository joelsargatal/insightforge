import os
import openai
from dotenv import load_dotenv
from agent.tools import sales_perf_monthly, sales_perf_quarterly, sales_perf_yearly, sales_product_region, sales_cust_segment, statistical_metrics, rag_search
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
import streamlit as st

# print(f"Current working directory: {os.getcwd()}")
# st.stop()

# Load environment variables from .env file
# load_dotenv()
load_dotenv(dotenv_path="/Users/scarbez-ai/Documents/Projects/_env/keys.env")

# Access the secrets safely
# api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")
# print( openai.api_key )

# Intercept standard greetings and return a generic message
def check_greeting(user_input: str) -> bool:
    """
    Check if the user's input is a greeting, like 'hi', 'hello', etc.
    Returns True if so, False otherwise.
    """
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    normalized = user_input.strip().lower()
    return any(normalized.startswith(greet) for greet in greetings)

# LLM
llm = ChatOpenAI(model="gpt-4o", openai_api_key=os.getenv("OPENAI_API_KEY"))

# Tool list
tools = [
    sales_perf_monthly,
    sales_perf_quarterly,
    sales_perf_yearly,
    sales_product_region,
    sales_cust_segment,
    statistical_metrics,
    rag_search
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

st.set_page_config(page_title="InsightForge", layout="wide")
st.title("🧩 InsightForge BI Assistant")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Ask about your sales data...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Intercept user greetings
    if check_greeting(user_input):
        response = (
            "👋 Hi! I can help you analyze your sales data or answer BI questions. "
            "Please ask me something specific about your data or business intelligence."
        )
    else:
        with st.spinner("Thinking..."):
            response = agent.invoke(user_input)
        response = response["output"]
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
