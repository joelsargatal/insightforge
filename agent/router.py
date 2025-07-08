# With this function we let route the user query to the agent flow or the conversational retrieval chain
#   based on a list of keywords. We use the conversational retrieval chain for BI concepts and practices
#   and the agent flow for sales data analysis, as the chain produces more detailed responses, better for
#   researching BI concepts and practices.
# If the user chooses a BI concept that is not included in the keywords to route the flow to the conversational
#   retrieval chain, the agent flow will be used, and the user will still receive valid responses, but more
#   concise ones.
# To-Do - Upgrade to use the LLM to decide, with broader rationale

def route_query(user_input: str) -> str:
    """
    Simple heuristic-based router to decide if input goes to RAG or data agent.
    Returns: "rag" or "agent"
    """
    input_lower = user_input.lower()

    # Keywords suggesting RAG (business intelligence, strategy, BI concepts)
    rag_keywords = [
        "what is", "define", "explain", "difference between", 
        "role of", "business intelligence", "bi", "ai", "olap", "oltp", "governance", "strategy", "maturity", "adoption"
    ]

    # Keywords suggesting BI metrics, sales, customer analysis
    agent_keywords = [
        "sales", "region", "product", "customer", "satisfaction", "age", "statistic", "segment", 
        "summarize", "analyze", "performance", "metric", "trend"
    ]

    if any(kw in input_lower for kw in rag_keywords):
        return "rag"
    if any(kw in input_lower for kw in agent_keywords):
        return "agent"

    # Default to agent (conservative fallback)
    return "agent"