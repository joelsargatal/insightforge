from langchain.tools import tool
from agent.data_handler import DataHandler
from rag.rag import vector_store, build_context_from_chunks, get_k_context_chunks
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize DataHandler once
data_handler = DataHandler("data/sales_data.csv")

def generate_monthly_sales_plot() -> plt.Figure:
    df_plot = pd.DataFrame(data_handler.get_monthly_sales_summary()["data"])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_plot["Month"], df_plot["Total Sales"], marker='o')
    ax.set_title("Monthly Sales Performance")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Sales")
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    return fig

def generate_quarterly_sales_plot() -> plt.Figure:
    df_plot = pd.DataFrame(data_handler.get_quarterly_sales_summary()["data"])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_plot["Quarter"], df_plot["Total Sales"], marker='o')
    ax.set_title("Quarterly Sales Performance")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Total Sales")
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    return fig


def generate_yearly_sales_plot() -> plt.Figure:
    df_plot = pd.DataFrame(data_handler.get_yearly_sales_summary()["data"])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_plot["Year"], df_plot["Total Sales"], marker='o')
    ax.set_title("Yearly Sales Performance")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Sales")
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    return fig

def generate_sales_product_region_plot() -> plt.Figure:
    plot_data = data_handler.get_product_region_sales_summary()
    df_plot = pd.DataFrame(data_handler.get_product_region_sales_summary()["data"])
    products = df_plot[plot_data["x"]].unique()
    regions = df_plot[plot_data["hue"]].unique()
    fig, ax = plt.subplots(figsize=(10, 6))

    for region in regions:
        region_data = df_plot[df_plot["Region"] == region]
        ax.bar(region_data[plot_data["x"]], region_data[plot_data["y"]], label=region)

    ax.set_xlabel(plot_data["x"])
    ax.set_ylabel(plot_data["y"])
    ax.set_title(plot_data["title"])
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title=plot_data["hue"])
    fig.tight_layout()
    return fig

def generate_sales_cust_segment_plot() -> plt.Figure:
    plot_data = data_handler.get_customer_segment_sales_summary()
    df_plot = pd.DataFrame(data_handler.get_customer_segment_sales_summary()["data"])

    fig, ax = plt.subplots(figsize=(10, 6))
    for gender in df_plot["Customer_Gender"].unique():
        subset = df_plot[df_plot["Customer_Gender"] == gender]
        ax.bar(subset["Age_Group"], subset["Total_Sales"], label=gender)

    ax.set_title(plot_data["title"])
    ax.set_xlabel(plot_data["x"])
    ax.set_ylabel(plot_data["y"])
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    fig.tight_layout()
    return fig

def generate_statistical_metrics_plot() -> plt.Figure:
    plot_data = data_handler.get_statistical_sales_summary()
    df_plot = pd.DataFrame(data_handler.get_statistical_sales_summary()["data"])

    # Convert y column to numeric, coercing errors (turns invalid into NaN)
    df_plot[plot_data["y"]] = pd.to_numeric(df_plot[plot_data["y"]], errors="coerce")
    # Drop any rows with NaN values in y
    df_plot = df_plot.dropna(subset=[plot_data["y"]])
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df_plot, x=plot_data["x"], y=plot_data["y"], hue=plot_data["hue"])

    ax.set_title(plot_data["title"])
    ax.set_xlabel(plot_data["x"])
    ax.set_ylabel(plot_data["y"])
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title=plot_data["hue"], bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    return fig

# Expose tools to the agent

# 1. Sales performance by time periods
# Separarte tools approach chosen to prevent having to filter words to get the time period or assume
#   the LLM guesses correctly 100% of the time

# Plotting-realted modifications
@tool
def sales_perf_monthly(query: str = ""):
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