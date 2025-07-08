from langchain.tools import tool
from agent.data_handler import DataHandler
from agent.rag import vector_store, build_context_from_chunks, get_k_context_chunks

# initialize your DataHandler once
data_handler = DataHandler("data/sales_data.csv")

# Expose tools to the agent

# 1. Sales performance by time periods
# Seperarte tools approach chosen to prevent having to filter words to get the time period or assume
#   the LLM guesses correctly 100% of the time
@tool
def sales_perf_monthly(query: str):
    """
    Summarizes sales performance by month.
    """
    return data_handler.sales_by_time_period(period="ME")

@tool
def sales_perf_quarterly(query: str):
    """
    Summarizes sales performance by quarter.
    """
    return data_handler.sales_by_time_period(period="QE")

@tool
def sales_perf_yearly(query: str):
    """
    Summarizes sales performance by year.
    """
    return data_handler.sales_by_time_period(period="YE")

# 2. Sales by product and region
@tool
def sales_product_region(query: str):
    """
    Provides total sales by product and region as a pivot table.
    """
    return data_handler.sales_by_product_region()

# 3. Sales by customer segment
@tool
def sales_cust_segment(query: str):
    """
    Segments customers by age group and gender, reporting total sales and average satisfaction.
    """
    return data_handler.sales_by_cust_segment()

# 4. Statistical metrics
@tool
def statistical_metrics(query: str):
    """
    Returns a summary of sales data using pandas describe.
    """
    return data_handler.statistical_metrics()

# RAG search tool
@tool
def rag_search(query: str):
    """
    Retrieve relevant BI knowledge from the embedded documents.
    """
    k_context_chunks = get_k_context_chunks()
    results = vector_store.similarity_search_with_score(query, k_context_chunks)
    context = build_context_from_chunks(results, k_context_chunks)

    # Compose a prompt for the agent
    wrapped_context = (
        f"The following business intelligence knowledge may help answer the question:\n\n"
        f"{context}\n\n"
        f"Please answer the question using this context."
    )

    return wrapped_context