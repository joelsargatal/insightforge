core_set = [
    {
        "query": "What is ETL in business intelligence?",
        "answer": "ETL stands for Extract, Transform, Load. It is the process of preparing data by extracting from sources, transforming it into a suitable format, and loading it into a data warehouse.",
        "prediction": None  # Will be filled during runtime
    },
    {
        "query": "What are the key metrics used in sales forecasting?",
        "answer": "Key metrics include historical sales data, seasonal trends, lead conversion rates, and average deal size.",
        "prediction": None
    },
    {
        "query": "How does big data analytics help in retail inventory management?",
        "answer": "It helps by predicting demand, optimizing stock levels, and reducing overstock or stockouts through real-time data.",
        "prediction": None
    },
    {
        "query": "What is a KPI dashboard used for?",
        "answer": "It visualizes key performance indicators to monitor business performance and support decision-making.",
        "prediction": None
    },
    {
        "query": "What is the difference between OLAP and OLTP?",
        "answer": "OLAP is used for analytical processing, OLTP for transactional processing.",
        "prediction": None
    }
]

rotation_set_1 = [
    {
        "query": "What are the components of a BI system?",
        "answer": "Key components include data sources, ETL processes, data warehouse, OLAP tools, and dashboards.",
        "prediction": None
    },
    {
        "query": "How is OLTP different from OLAP?",
        "answer": "OLTP systems are optimized for transactional tasks; OLAP is used for complex queries and analysis.",
        "prediction": None
    },
    {
        "query": "Explain ETL in business intelligence.",
        "answer": "ETL stands for Extract, Transform, Load — a process to move and prepare data for analysis.",
        "prediction": None
    },
    {
        "query": "List benefits of a BI platform.",
        "answer": "Improved decision-making, real-time insights, and data-driven culture.",
        "prediction": None
    },
    {
        "query": "What is data governance in BI?",
        "answer": "Data governance ensures data quality, security, and compliance in BI systems.",
        "prediction": None
    }
]

rotation_set_2 = [
    {
        "query": "What’s the purpose of a data warehouse?",
        "answer": "A centralized repository to store and manage data for analysis and reporting.",
        "prediction": None
    },
    {
        "query": "How can AI improve BI dashboards?",
        "answer": "AI can automate insights, personalize views, and detect anomalies in dashboards.",
        "prediction": None
    },
    {
        "query": "Describe a time-series analysis.",
        "answer": "It analyzes data points collected or recorded at specific time intervals.",
        "prediction": None
    },
    {
        "query": "What is a KPI?",
        "answer": "A KPI is a Key Performance Indicator — a metric used to measure business success.",
        "prediction": None
    },
    {
        "query": "How does Walmart use data analytics?",
        "answer": "Walmart uses analytics to optimize logistics, manage inventory, and personalize marketing.",
        "prediction": None
    }
]

# Collect all sets into a dictionary
qa_sets = {
    "core": core_set,
    "rotation_1": rotation_set_1,
    "rotation_2": rotation_set_2
}