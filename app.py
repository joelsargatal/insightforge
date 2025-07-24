import os
import openai
from dotenv import load_dotenv
from pprint import pprint
from agent.tools import sales_perf_monthly, sales_perf_quarterly, sales_perf_yearly, sales_product_region, sales_cust_segment, statistical_metrics, rag_search
from agent.tools import generate_monthly_sales_plot, generate_quarterly_sales_plot, generate_yearly_sales_plot, generate_sales_product_region_plot, generate_sales_cust_segment_plot, generate_statistical_metrics_plot
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from agent.rag_chat import qa_chain, get_memory
from utils.monitoring import rag_callbacks, agent_callbacks, token_tracker, rag_oa_cb_handler, agent_oa_cb_handler
import streamlit as st



import sys

# Optional: import early to avoid circular import issues
from evaluation.qa_eval import run_evaluation  # Replace with correct module name
from evaluation.qa_eval_sets import qa_sets   # If sets are in a separate module

if len(sys.argv) > 1:
    eval_set_name = sys.argv[1]
    if eval_set_name in qa_sets:
        examples = qa_sets[eval_set_name]
        run_evaluation(qa_set_name=eval_set_name)
        # run_evaluation(examples, qa_set_name=eval_set_name)
    else:
        print(f"⚠️ Unknown evaluation set: '{eval_set_name}'")
    st.stop()




# Load environment variables from .env file
load_dotenv(dotenv_path="/Users/scarbez-ai/Documents/Projects/_env/keys.env")

# Set the OpenAI key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Utility function for displaying data structures in a more readable format
def pretty_print(data):
    """Nicely prints any Python data structure."""
    pprint(data, indent=2, width=100, compact=False)

# Classify user input using LLM
def classify_input(user_input: str) -> str:
    """
    Classifies user input as:
    - 'bi_query': A Business Intelligence-related data question
    - 'sales_analysis': Request to perform sales data analysis
    - 'plot_request': Request to generate a visual or plot of the data
    - 'greeting': Social or polite greeting
    - 'off_topic': Unrelated to BI or sales data
    - 'unclear': Can't classify
    """
    prompt = f"""
    You are an assistant that classifies user input related to a business intelligence assistant that can analyze sales data and business documents.

    Classify each query as one of:
    - bi_query: Asking about business intelligence (e.g. ETL, KPIs), insights, stats, etc.
    - sales_analysis: Request to perform sales data analysis.
    - plot_request: Explicitly asking for charts or graphs.
    - greeting: A social greeting.
    - off_topic: Clearly not related to business, sales, or data.
    - unclear: Cannot be confidently categorized.

    Examples:

    User: "Hello, how are you?"
    Classification: greeting

    User: "What are the key components of a BI system?"
    Classification: bi_query

    User: "Explain the difference between OLAP and OLTP."
    Classification: bi_query

    User: "What is ETL in business intelligence?"
    Classification: bi_query

    User: "List some benefits of implementing a BI platform."
    Classification: bi_query

    User: "How can AI enable business model innovation?"
    Classification: bi_query

    User: "What challenges exist in adopting AI for business model innovation?"
    Classification: bi_query

    User: "Give examples of AI-driven value propositions."
    Classification: bi_query

    User: "How is IoT data used in time-series forecasting?"
    Classification: bi_query

    User: "Describe a machine learning pipeline for predicting air quality using IoT."
    Classification: bi_query

    User: "What are the limitations of IoT data for ML predictions?"
    Classification: bi_query

    User: "How did Walmart use big data analytics to improve operations?"
    Classification: bi_query

    User: "What technologies did Walmart adopt to handle large-scale sales data?"
    Classification: bi_query

    User: "Summarize Walmart’s approach to customer segmentation."
    Classification: bi_query

    User: "Compare the benefits of business intelligence with those of big data analytics."
    Classification: bi_query

    User: "What role does data governance play in BI and AI adoption?"
    Classification: bi_query

    User: "Explain how time-series prediction techniques could integrate with a BI dashboard."
    Classification: bi_query

    User: "Analyze sales data by month."
    Classification: sales_analysis

    User: "Analyze sales data by quarter."
    Classification: sales_analysis

    User: "Analyze sales data by year."
    Classification: sales_analysis

    User: "Analyze sales data by product and region."
    Classification: sales_analysis

    User: "Analyze sales data by customer segment."
    Classification: sales_analysis

    User: "Provide statistical metrics of the sales data."
    Classification: sales_analysis

    User: "Do statistical analysis on the sales data."
    Classification: sales_analysis

    User: "Can you show me a graph of sales this month?"
    Classification: plot_request

    User: "Analyze sales data by month and plot the results in a graph."
    Classification: plot_request

    User: "Analyze sales data by quarter and plot the results in a graph."
    Classification: plot_request

    User: "Analyze sales data by year and plot the results in a graph."
    Classification: plot_request

    User: "What’s your favorite color?"
    Classification: off_topic

    User: "Sales"
    Classification: unclear

    User: "{user_input}"
    Classification:
    """.strip()
    try:
        classification = llm.predict(prompt).strip().lower()
        return classification
    except Exception as e:
        print("Classification error:", e)
        return "unclear"

# Get shared memory, defined in RAG chain source
memory = get_memory()

# LLM
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    callbacks=agent_callbacks,
    verbose=True
)

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
    memory=memory,
    verbose=True,
    callbacks=agent_callbacks,
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
    intent = "error"
    assistant_type = ""

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    intent = classify_input(user_input)
    print(f"[Classifier] Intent: {intent}")

    # Intercept user greetings
    if intent == "greeting":
        response = "👋 Hi! I can help you analyze your sales data or answer BI-related questions."
    
    elif intent == "off_topic":
        response = "🔍 I’m focused on Business Intelligence and your data. Please ask me something related to that."

    elif intent == "unclear":
        response = "❓ I couldn’t understand your question. Can you rephrase it in the context of your business or data?"

    elif intent == "bi_query":
        assistant_type = " (RAG): "
        print("[Router] Routed to: RAG")
        print("Thinking...")
        with st.spinner("Thinking..."):
            response = qa_chain.invoke({"question": user_input}, config={"callbacks": rag_callbacks})
            token_tracker["rag"].update(rag_oa_cb_handler)
            # response = qa_chain.invoke({"question": user_input})
            response = response["output"]
        print("Assistant" + assistant_type + ": ", response)
    elif intent in ["plot_request", "sales_analysis"]:
        assistant_type = " (Agent): "
        print("[Router] Routed to: Agent")
        print("Thinking...")
        adj_user_input = user_input
        with st.spinner("Thinking..."):
            if intent == "plot_request":
                if "month" in adj_user_input.lower():
                    fig = generate_monthly_sales_plot()
                elif "quarter" in adj_user_input.lower():
                    fig = generate_quarterly_sales_plot()
                elif "year" in adj_user_input.lower():
                    fig = generate_yearly_sales_plot()
                elif "product" in adj_user_input.lower():
                    fig = generate_sales_product_region_plot()
                elif "segment" in adj_user_input.lower():
                    fig = generate_sales_cust_segment_plot()
                elif "metrics" in adj_user_input.lower():
                    fig = generate_statistical_metrics_plot()
                else:
                    raise ValueError("Could not determine plot type from query.")
                print("📈 Plotting...")
                st.pyplot(fig)
                adj_user_input = user_input + ". Ignore the plotting ask. Do not generate code for plotting."
            response = agent.invoke(adj_user_input, config={"callbacks": agent_callbacks})
            token_tracker["agent"].update(agent_oa_cb_handler)
            # response = agent.invoke(adj_user_input)
            response = response["output"]
        print("Assistant" + assistant_type + ": ", response)
    elif not intent or intent == "error":
        assistant_type = " (Error): "
        response = "🥴 Something is wrong..."
        print("Assistant" + assistant_type + ": ", response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

    # Enable to debug memory issues
    # print("\nMemory:")
    # pretty_print(memory.chat_memory.messages)
    # print("\n")

    print("📊 Session Token Summary:")
    print("Agent:", token_tracker["agent"])
    agent_token_cost = token_tracker["agent"].estimate_cost_usd('gpt-4o')
    print(f"Agent Token Cost (USD): ${agent_token_cost:.6f}")
    print("RAG:", token_tracker["rag"])
    rag_token_cost = token_tracker["rag"].estimate_cost_usd('gpt-4o')
    print(f"RAG Token Cost (USD): ${rag_token_cost:.6f}")
    total_tokens = token_tracker["agent"].total_tokens + token_tracker["rag"].total_tokens
    print("Total Tokens:", total_tokens)
    total_cost = agent_token_cost + rag_token_cost
    print(f"Total Cost (USD): ${total_cost:.6f}")