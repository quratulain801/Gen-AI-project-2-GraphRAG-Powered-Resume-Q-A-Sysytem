GraphRAG-Powered Resume Q&A System

This project implements an AI-powered Resume Question-Answering system using GraphRAG (Graph Retrieval-Augmented Generation) and Neo4j Knowledge Graph.

The system allows recruiters to ask natural language questions (e.g., “Who has 3+ years of Python experience?”) and get structured, explainable answers based on stored resumes.

🚀 Features

Automated resume ingestion (PDF/DOCX → text extraction).

Chunking & Embedding of resumes using LLMs.

Neo4j Knowledge Graph to store candidates, skills, education, and experience.

GraphRAG-powered Q&A: Converts recruiter questions into Cypher queries.

Explainable results (why a candidate was selected).

Supports semantic search and structured queries.

📂 Project Structure
├── main.py              # Main entry point
├── requirements.txt     # Dependencies
├── .env                 # Environment variables
├── README.md            # Documentation
└── data/                # Resume files (PDF/DOCX/TXT)

⚙️ Installation

Clone repo

git clone https://github.com/your-username/graphrag-resume-qa.git
cd graphrag-resume-qa


Install dependencies

pip install -r requirements.txt


Setup environment variables
Create a .env file:

OPENAI_API_KEY=your_api_key
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

▶️ Usage

Run the system:

python main.py


Ask questions such as:

“Find candidates with Python and SQL skills.”

“Which candidates are the best match for Data Analyst role?”

“List candidates with more than 3 years of experience in Machine Learning.”

The system will:

Convert question → Cypher query.

Retrieve results from Neo4j.

Generate natural language answer for recruiter.

📊 Example Output

Query:
“Who has more than 3 years of Python experience?”

Answer:

Ali Khan – 5 years Python
Sara Ahmed – 4 years Python


Query:
“Best match for Data Analyst role?”

Answer:

Ali Khan – Match Score 92%
Sara Ahmed – Match Score 78%

📈 Workflow (High-Level)

Resume Upload → Text Extraction

Embedding + Chunking

Neo4j Knowledge Graph storage

GraphRAG Retrieval

LLM Q&A Module

Answer shown to Recruiter

🔮 Future Improvements

Integration with ATS (Applicant Tracking Systems).

Multi-format support (DOCX, TXT, LinkedIn profiles).

Real-time dashboard for recruiters.

Graph embeddings for advanced similarity search.

🛡️ Security & Privacy

PII anonymization (candidates stored with IDs).

Encrypted credentials (via .env).

Role-based access for recruiters.
