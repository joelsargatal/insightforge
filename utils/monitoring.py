from langchain.callbacks import StdOutCallbackHandler, OpenAICallbackHandler

# Token Tracker to hold cumulative stats
class TokenTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    def update(self, handler: OpenAICallbackHandler):
        self.total_tokens += handler.total_tokens
        self.prompt_tokens += handler.prompt_tokens
        self.completion_tokens += handler.completion_tokens
        self.calls += 1

    def report(self):
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens
        }

    def estimate_cost_usd(self, model="gpt-4o"):
        # Prices per 1M tokens as of July 2025
        prices = {
            "gpt-4o": {
                "input": 0.005,     # $5.00 / 1M
                "output": 0.02      # $20.00 / 1M
            },
            "gpt-4": {
                "input": 0.03,
                "output": 0.06
            },
            "gpt-3.5": {
                "input": 0.001,
                "output": 0.002
            }
        }

        p = prices.get(model, prices["gpt-4o"])
        cost = (
            (self.prompt_tokens / 1_000_000) * p["input"] +
            (self.completion_tokens / 1_000_000) * p["output"]
        )
        # return float(f"{cost:.6f}")
        return round(cost, 6)

    def __str__(self):
        return (
            f"🧮 Token Usage:\n"
            f"  Calls: {self.calls}\n"
            f"  Prompt Tokens: {self.prompt_tokens}\n"
            f"  Completion Tokens: {self.completion_tokens}\n"
            f"  Total Tokens: {self.total_tokens}"
        )

# Set up callback handler for tracing
# Handlers (shared instances)
stdout_handler = StdOutCallbackHandler()
rag_oa_cb_handler = OpenAICallbackHandler()
agent_oa_cb_handler = OpenAICallbackHandler()

rag_callbacks = [stdout_handler, rag_oa_cb_handler]
agent_callbacks = [stdout_handler, agent_oa_cb_handler]

token_tracker = {
    "rag": TokenTracker(),
    "agent": TokenTracker()
}