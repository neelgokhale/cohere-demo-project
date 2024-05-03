import os

from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_cohere import ChatCohere

from dotenv import load_dotenv
load_dotenv()

llm = ChatCohere(model='command-r-plus', temperature=0, cohere_api_key=os.getenv("COHERE_API_KEY"))

graph = Neo4jGraph()

llm_transformer = LLMGraphTransformer(llm=llm)

from langchain_core.documents import Document

text = """
Marie Curie, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.
She was, in 1906, the first woman to become a professor at the University of Paris.
"""

documents = [Document(page_content=text)]

graph_documents = llm_transformer.convert_to_graph_documents(documents)

for node in graph_documents[0].nodes:
    print(node)

for rel in graph_documents[0].relationships:
    print(rel)


graph.add_graph_documents(graph_documents)