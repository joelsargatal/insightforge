from langchain.tools import tool
from agent.data_handler import DataHandler
from agent.rag import vector_store, build_context_from_chunks, get_k_context_chunks
import pandas as pd
import matplotlib.pyplot as plt

# initialize your DataHandler once
data_handler = DataHandler("data/sales_data.csv")

def wants_plot(user_input: str) -> bool:
    """
    Check if the user asks for a plot in their query.
    Returns: True if input suggests a plot, False otherwise
    """

    # Keywords suggesting a plot
    plot_keywords = [
        "plot", "chart", "graph", "diagram", "visualization", "visualize", "visual", "display", "show", "illustrate"
    ]

    normalised = user_input.strip().lower()
    return any(kw in normalised for kw in plot_keywords)


def generate_monthly_sales_plot() -> plt.Figure:
    plot_data = data_handler.get_monthly_sales_summary()
    df_plot = pd.DataFrame(data_handler.get_monthly_sales_summary()["data"])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_plot["Month"], df_plot["Total Sales"], marker='o')
    ax.set_title("Monthly Sales Performance")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Sales")
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    return fig
# Expose tools to the agent

# 1. Sales performance by time periods
# Seperarte tools approach chosen to prevent having to filter words to get the time period or assume
#   the LLM guesses correctly 100% of the time

# Plotting-realted modifications
@tool
def sales_perf_monthly(query: str = ""):
    """
    Summarizes sales performance by month.
    """
    return data_handler.sales_by_time_period(period="ME")

# @tool
# def sales_perf_monthly(query: str):
#     """
#     Summarizes sales performance by month.
#     """
#     return data_handler.sales_by_time_period(period="ME")

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