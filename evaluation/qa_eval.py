from langchain.evaluation.qa import QAEvalChain
from langchain_openai import ChatOpenAI
from evaluation.qa_eval_sets import qa_sets
import random

# Setup LLM and evaluation chain
llm = ChatOpenAI(model="gpt-4o")
eval_chain = QAEvalChain.from_llm(llm)

def initialize_predictions(examples):
    for ex in examples:
        if "prediction" not in ex:
            result = qa_chain.invoke({"question": ex["query"]})
            ex["prediction"] = result["output"]

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
    predictions = [{"query": ex["query"], "result": ex.get("prediction", ex["answer"])} for ex in examples]
    references = [{"query": ex["query"], "answer": ex["answer"]} for ex in examples]

    graded_outputs = eval_chain.evaluate(examples=references, predictions=predictions)

    # Display results
    for i, result in enumerate(graded_outputs):
        print(f"\n🔹 Q: {examples[i]['query']}")
        print(f"✅ A: {examples[i]['answer']}")
        print(f"🤖 P: {predictions[i]['result']}")
        print(f"🧠 Grade: {result.get('results')}")
        print(f"📝 Explanation: {result.get('reasoning', 'N/A')}")

    return graded_outputs