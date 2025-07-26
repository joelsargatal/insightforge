import os
import openai
from dotenv import load_dotenv
import random
from rag.rag_chat import qa_chain
from langchain.evaluation.qa import QAEvalChain
from langchain_openai import ChatOpenAI
from evaluation.qa_eval_sets import qa_sets
import matplotlib.pyplot as plt
import streamlit as st

# Load environment variables from .env file
load_dotenv(dotenv_path="/Users/scarbez-ai/Documents/Projects/_env/keys.env")

# Set the OpenAI key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Setup LLM and evaluation chain
llm = ChatOpenAI(model="gpt-4o")
eval_chain = QAEvalChain.from_llm(llm)

def plot_eval_results(results):
    counts = {"CORRECT": 0, "INCORRECT": 0, "Other": 0}
    for res in results:
        grade = res.get("results", "").upper()
        if grade in counts:
            counts[grade] += 1
        else:
            counts["Other"] += 1

    plt.bar(counts.keys(), counts.values(), color=["green", "red", "gray"])
    plt.title("QAEvalChain Results")
    plt.ylabel("Count")
    # plt.show()
    st.pyplot(plt.gcf())

def initialize_predictions(examples):
    for i, ex in enumerate(examples):
        if not ex.get("prediction"):
            print(f"🔍 Generating prediction for Q{i+1}: {ex['query']}")
            st.markdown(f"🔍 Generating prediction for Q{i+1}: {ex['query']}")
            try:
                result = qa_chain.invoke({"question": ex["query"]})
                ex["prediction"] = result["output"]
                # ex["prediction"] = result["results"]
            except Exception as e:
                ex["prediction"] = "ERROR"
                print(f"⚠️ Failed to generate prediction: {e}")
                st.markdown(f"⚠️ Failed to generate prediction: {e}")

def run_evaluation(qa_set_name="core", sample_size=None):
    print(f"📊 Evaluating set: {qa_set_name}")

    examples = qa_sets.get(qa_set_name, [])
    initialize_predictions(examples)

    if not examples:
        print(f"⚠️ No examples found for set: {qa_set_name}")
        return []

    if sample_size:
        examples = random.sample(examples, min(sample_size, len(examples)))

    # Evaluate using correct answer vs prediction (for now we mock predictions)
    predictions = [{"query": ex["query"], "result": ex["prediction"]} for ex in examples]
    references = [{"query": ex["query"], "answer": ex["answer"]} for ex in examples]

    graded_outputs = eval_chain.evaluate(examples=references, predictions=predictions)

    # Display results
    for i, result in enumerate(graded_outputs):
        q = examples[i]['query']
        a = examples[i]['answer']
        p = predictions[i]['result']
        g = result.get('results')
        r = result.get('reasoning', 'N/A')
        print(f"\n🔹 Q: {q}")
        print(f"✅ A: {a}")
        print(f"🤖 P: {p}")
        print(f"🧠 Grade: {g}")
        print(f"📝 Explanation: {r}")
        st.markdown(f"""
        **🔹 Q:** {q}  
        **✅ A:** {a}  
        **🤖 P:** {p if p else "None"}  
        **🧠 Grade:** `{g}`  
        **📝 Explanation:** {r}
        """)

    plot_eval_results(graded_outputs)

    return graded_outputs