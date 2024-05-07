import os
import sys
import logging

import warnings
warnings.filterwarnings('ignore')

from llama_index.llms.cohere import Cohere
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.core import Settings, PromptTemplate

from llama_index.core import KnowledgeGraphIndex, SimpleDirectoryReader, ServiceContext
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.graph_stores.neo4j import Neo4jGraphStore

from llama_index.core.schema import MetadataMode

from dotenv import load_dotenv
load_dotenv()

INDEX_PATH = "data/index/med_index"
DATA_DIR = "data/med"

template = (
    "You are a top-class medical knowledge extraction tool used to retrieve"
    "important medical entities, relationships and extracting information in structured"
    "formats to build a knowledge graph."
    "You will be fed with various types of medical and health-related reports and"
    "documentation, which you will have to intrepret into the structure of a graph."
    "Each node should have an id"
    "- **Nodes** represent entities and concepts.\n"
    "- The aim is to achieve simplicity and clarity in the knowledge graph, making it\n"
    "accessible for a vast audience.\n"
    "A node should ALWAYS have an 'id' and a 'type' so NEVER generate one without these."
    "## 2. Labeling Nodes\n"
    "- **Consistency**: Ensure you use available types for node labels.\n"
    "Ensure you use basic or elementary types for node labels.\n"
    "- For example, when you identify an entity representing a person, "
    "always label it as **'person'**. Avoid using more specific terms "
    "like 'mathematician' or 'scientist'"
)

class LIGraphRetriever(object):
    def __init__(self) -> None:
        # define llm and embeddings

        llm = Cohere(temperature=0, model='command-r-plus')
        embeddings = CohereEmbedding()
        
        Settings.llm = llm
        Settings.embed_model = embeddings

        # setup neo4j
        graph = Neo4jGraphStore(
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            url=os.getenv("NEO4J_URI"),
            database=os.getenv("NEO4J_DB"),
            nodel_label="entity"
        )

        if len(os.listdir(INDEX_PATH)) == 0:
            # load documents from data directory
            docs = SimpleDirectoryReader(
                DATA_DIR
            ).load_data()

            storage_context = StorageContext.from_defaults(graph_store=graph)

            # setup index
            edge_types = ['relationship']
            tags = ['entity']
            
            index = KnowledgeGraphIndex.from_documents(
                docs,
                chunk_size=500,
                chunk_overlap=100,
                storage_context=storage_context,
                max_triplets_per_chunk=15,
                include_embeddings=True,
                graph_store_query_depth=6,
                tags=tags,
                edge_types=edge_types,
            )
            
            # store index
            index.storage_context.persist(persist_dir=INDEX_PATH)
            
        else:
            storage_context = StorageContext.from_defaults(persist_dir=INDEX_PATH)
            index = load_index_from_storage(storage_context=storage_context)

        self.ret = index.as_retriever(include_text=False,
                                      embedding_model="hybrid",
                                      use_global_node_triplets=True,
                                      graph_traversal_depth=100,
                                      similarity_top_k = 5
                                     )
    
    def get_relevant_chains(self, query: str) -> list[str]:

        raw_chains = self.ret.retrieve(query)[0].metadata['kg_rel_texts']
        
        chains = []

        for chain in raw_chains:
            chain = chain.replace("(", "").replace(")", "").replace("'", "")
            chain = " ".join(chain.split(", "))
            chains.append(chain)

        return chains
