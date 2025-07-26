# 📊 InsightForge: AI-Powered Business Intelligence Assistant

**InsightForge** is an intelligent assistant for sales data analysis and business intelligence. Built with OpenAI's GPT-4o, LangChain, and Streamlit, it combines Retrieval-Augmented Generation (RAG), prompt routing, agent tools, and visualization to deliver meaningful insights from structured datasets.

---

## 🚀 Features

- 🔍 **Intent Classification** – Classifies user queries as sales, BI, charting, etc.
- 🧠 **RAG Q&A Engine** – Uses a vector database and LLM to answer BI-related questions.
- 🤖 **Agent-Based Routing** – Directs queries to specialized tools based on intent and query.
- 📈 **Plotting Tools** – Supports monthly, quarterly, yearly, and segmented visualizations.
- ✅ **LLM Evaluation** – Integrated `QAEvalChain` for automated answer grading.

---

## 🧪 Quick Start

### ✅ Run the Assistant

```bash
streamlit run app.py

🧪 Run Evaluation Mode (Optional)

streamlit run app.py core

Replace core with rotation_1 or rotation_2 to evaluate different Q&A sets.

⸻

📁 Directory Structure

└── _env/
	└── keys.env                        # Environment variables (OpenAI key, not committed)
└── insightforge/
	├── app.py                          # Streamlit UI and application entry point
	├── README.md                       # You are here!
	├── requirements.txt                # Required Python packages
	└── agent/
		├── data_handler.py             # Data analysis functions
		└── tools.py                    # Agent tools for analysis and plotting
	└── archive/
		└── router.py                   # Agent-RAG routing logic - deprecated
	└── data/
		└── sales_data.csv              # Source sales data
	└── db/                             # Vector database cache directory. Created at runtime - not committed
		└── …                           
	└── docs/                           # BI PDFs and metadata
		├── AI-Driven-Business-Model-Innovation.pdf
		├── Business-Intelligence-Concepts-and-Approaches.pdf
		├── metadata.json               # Document metadata. Created at runtime - not committed
		├── Time-Series-Data-Prediction-Using-IoT-and-ML.pdf
		└── Walmart_s-Sales-Data-Analysis.pdf
	└── evaluation/
		├── qa_eval.py                  # Evaluation logic using QAEvalChain
		└── qa_eval_sets.py             # Sample Q&A sets for evaluation
	└── notebooks/
		└── InsightForge_Dev_Notebook.ipynb
	└── rag/
		├── rag.py                      # RAG logic
		└── rag_chat.py                 # RAG chat interface
	└── utils/
		└── monitoring.py               # Token usage and cost tracking


⚙️ Setup Instructions
	1.	Install dependencies

pip install -r requirements.txt


	2.	Set your OpenAI API Key
Create a .env file in the root directory with:

OPENAI_API_KEY=your-key-here


	3.	Prepare your data
Place your sales CSV file as data/sales_data.csv. The file should include at least:
	•	Date
	•	Sales
	•	Product
	•	Region
	•	Customer_Gender
	•	Customer_Age
	•	Customer_Satisfaction

	4.	Prepare your documents
Place your BI documents in the docs/ directory. The application is going to load them, add a metadata file for tracking document additions, removals, or new versions, ingest them, classify them, and create a vector database.


📄 Documentation

A full project walkthrough, design rationale, and development highlights are available in the accompanying documentation PDF.


🧠 Built With
	•	OpenAI GPT-4o
	•	LangChain
	•	Streamlit
	•	Matplotlib
	•	Pandas


📬 License

For academic and evaluation use only. All trademarks belong to their respective owners.


Let me know if you'd like a second version that includes screenshots, badges, or links to example outputs.