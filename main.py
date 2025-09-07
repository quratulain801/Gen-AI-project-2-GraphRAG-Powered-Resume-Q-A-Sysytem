import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_core.output_parsers import StrOutputParser
from neo4j import GraphDatabase

load_dotenv()

# --- Initialize Neo4j Connection ---
URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

def init_neo4j_connection():
    graph = Neo4jGraph(url=URI, username=USERNAME, password=PASSWORD)
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    return graph, driver

graph, driver = init_neo4j_connection()

# --- Get Schema ---
def get_schema(graph):
    return graph.get_schema if graph else ""

schema = get_schema(graph)

# --- Initialize Graph QA Chain ---
def init_qa_chain(graph):
    template = """
    Task: Generate a Cypher statement to query the graph database.
    You will be given a rephrased query.
    Generate a cypher statement to answer the rephrased query.

    Instructions:
    1. Search for nodes and their relationships
    2. Return relevant information about the queried entities
    3. Don't restrict to specific labels, search across all nodes
    4. Use CONTAINS or other fuzzy matching when appropriate

    schema:
    {schema}

    Rephrased Query: {query}

    Cypher Statement:
    """
    question_prompt = PromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4o")
    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        cypher_prompt=question_prompt,
        verbose=True,
        allow_dangerous_requests=True,
        return_intermediate_steps=True,
    )
    return chain

qa = init_qa_chain(graph)

# --- Extract nodes and relationships ---
def relationship_to_string(relationship):
    node1, node2 = relationship.nodes
    label1 = list(node1.labels)[0] if node1.labels else ""
    label2 = list(node2.labels)[0] if node2.labels else ""
    name1 = node1.get("name", "Unknown")
    name2 = node2.get("name", "Unknown")
    rel_type = relationship.type
    return f'(:{label1} {{name: "{name1}"}})-[:{rel_type}]->(:{label2} {{name: "{name2}"}})'

def get_all_nodes_and_relationships(driver):
    nodes_list = []
    relationships_list = []
    if not driver:
        return nodes_list, relationships_list

    with driver.session() as session:
        base_query = """
        MATCH (n:Entity)
        OPTIONAL MATCH (n)-[r]-(m:Entity)
        RETURN DISTINCT n, r, m
        """
        result = session.run(base_query)
        for record in result:
            if record.get("n") and record["n"].get("name"):
                nodes_list.append(record["n"]["name"])
            if record.get("m") and record["m"].get("name"):
                nodes_list.append(record["m"]["name"])
            if record.get("r") is not None:
                relationships_list.append(relationship_to_string(record["r"]))
    return list(set(nodes_list)), list(set(relationships_list))

nodes_list, relationships_list = get_all_nodes_and_relationships(driver)

# --- Rephrase Query ---
def rephrase_query_chain(query, nodes, relationships):
    if not query:
        return None
    rephrase_template = """You are a highly skilled assistant that specializes in rephrasing user queries using the list of nodes and the list of relationships in the graph database.

List of nodes: {nodes}
List of relationships: {relationships}
Query: {query}

Rephrased Query:
"""
    llm = ChatOpenAI(model="gpt-4o")
    prompt = PromptTemplate.from_template(rephrase_template)
    chain = prompt | llm
    return chain.invoke({"nodes": nodes, "relationships": relationships, "query": query})

# --- Example query ---
query = "which candidate has git skills"
rephrased_query = rephrase_query_chain(query, nodes_list, relationships_list)

# --- Execute Cypher Query ---
final_result = qa.invoke({"query": rephrased_query, "schema": schema})

# --- Combine Answer ---
combined_answer_template = """
You will be provided with a user query, schema of knowledge graph and result from
cypher query chain.

Your task is to draft a final answer based on the result and the user query. Before drafting 
the final response use context provided by the result.
Schema: {schema}
Result: {final_result}
Combined Answer:
"""
prompt = PromptTemplate.from_template(combined_answer_template)
llm = ChatOpenAI(model="gpt-4o")
chain = prompt | llm
combined_result = chain.invoke({"final_result": final_result, "schema": schema})

print(combined_result)
