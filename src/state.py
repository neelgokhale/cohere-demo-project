# ../src/state.py

import os

import warnings
warnings.filterwarnings('ignore')

from typing_extensions import TypedDict
from typing import List
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import PubMedRetriever
from langgraph.graph import END, StateGraph

import certifi
import ssl
import urllib
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())


from retriever import LIGraphRetriever

from dotenv import load_dotenv
load_dotenv()


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of knowledge chains and/or documents retrieved
        is_invalid: whether input question is invalid (PHI / irrelevance)
    """

    question: str
    generation: str
    web_search: str
    documents: List[str]
    is_invalid: str
    

# GRADER

grader_llm = ChatCohere(model='command-r-plus', temperature=0, cohere_api_key=os.getenv("COHERE_API_KEY"))

class GradeGraphChains(BaseModel):
    """Binary score for relevant graph chains in the KG"""
    
    binary_score: str = Field(
        description="Determine if a retrieved knowledge graph chain is relevant to the query, 'yes' or 'no'"
    )
    

GRADER_SYS = (
    "You are a grader assessing relevance of a retrieved document to a user question. " 
    "If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. "
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. "
)

grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", GRADER_SYS),
        ("human", "Retrieved chains: \n\n {chain} \n\n User question: {question}")
    ]
)

grader_output = grader_llm.with_structured_output(GradeGraphChains)

grader = grader_prompt | grader_output

question = "What is a benign tumor"

doc = Document(page_content="Tumors Associated with Chronic inflammation")

# PROMPT RE-WRITER
    
REWRITER_SYS = (
    "You a question re-writer that converts an input question a relevant search topic"
    "for searching research material on PubMed. Only produce a 1 line answer"
    "Look at the input and try to reason about the underlying sematic intent / meaning."
    "An example of this would be:\n"
    "Initial question: What are the early-signs of colorectal cancer"
    "Early signs for colorectal cancer"
)

rewriter_llm = ChatCohere(model='command-r-plus', temperature=0, cohere_api_key=os.getenv("COHERE_API_KEY"))

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", REWRITER_SYS),
        (
            "human", 
            "Here is the initial question: \n\n {question} \n Extract all relevant entities that can be used to reference PubMed, a medical knowledge base."
        )
    ]
)

rewriter = rewrite_prompt | rewriter_llm | StrOutputParser()

# RAG

RAG_PROMPT = (
    "You are a top medical assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "The context provided contains a mixture of clinical observations retrieved from "
    "Medical log files and patient charts and also may include retrieved research documents "
    "from PubMed, a medical research knowledge base. Try and use the context judiciously. "
    "Always stick to the facts and provide factual evidence based on the context when explaining your answer. "
    "If possible, be succint and concise and avoid irrelevant information. "
    "If you don't know the answer, just say that you don't know.\n"
    "Question: {question} \n"
    "Context: {context} \n"
    "Answer: "
)

rag_llm = ChatCohere(model='command-r-plus', temperature=0, cohere_api_key=os.getenv("COHERE_API_KEY"))

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
rag_generator = rag_prompt | rag_llm | StrOutputParser()

# PHI CHECKER

class PHICheckerResult(BaseModel):
    """Based on the presence of PHI outputs a 'YES' or 'NO'"""
    
    is_invalid: str = Field(
        description="reply with a 'yes' or 'no'. Determine if a question is irrelevant to any medical context or if a question asks for personal health records"
    )

CHECKER_SYSTEM = (
    "You are a top class medical private health information (PHI) checker. You will determine if "
    "the input question either:\n"
    " 1. asks for information irrelevant to any medical or health topics\n"
    " 2. violates any HIPAA data privacy standards by identifying if it contains "
    "any PHI from the below examples: \n"
    "1. Names;\n"
    "2. All geographical subdivisions smaller than a State, including street address, city, county, precinct, zip code, and their equivalent geocodes, except for the initial three digits of a zip code, if according to the current publicly available data from the Bureau of the Census: (1) The geographic unit formed by combining all zip codes with the same three initial digits contains more than 20,000 people; and (2) The initial three digits of a zip code for all such geographic units containing 20,000 or fewer people is changed to 000.\n"
    "3. All elements of dates (except year) for dates directly related to an individual, including birth date, admission date, discharge date, date of death; and all ages over 89 and all elements of dates (including year) indicative of such age, except that such ages and elements may be aggregated into a single category of age 90 or older;\n"
    "4. Phone numbers;\n"
    "5. Fax numbers;\n"
    "6. Electronic mail addresses;\n"
    "7. Social Security numbers;\n"
    "8. Medical record numbers;\n"
    "9. Health plan beneficiary numbers;\n"
    "10. Account numbers;\n"
    "11. Certificate/license numbers;\n"
    "12. Vehicle identifiers and serial numbers, including license plate numbers;\n"
    "13. Device identifiers and serial numbers;\n"
    "14. Web Universal Resource Locators (URLs);\n"
    "15. Internet Protocol (IP) address numbers;\n"
    "16. Biometric identifiers, including finger and voice prints;\n"
    "17. Full face photographic images and any comparable images; and\n"
    "18. Any other unique identifying number, characteristic, or code (note this does not mean the unique code assigned by the investigator to code the data)"
)

checker_llm = ChatCohere(model='command-r-plus', temperature=0, cohere_api_key=os.getenv("COHERE_API_KEY"))
checker_llm = checker_llm.with_structured_output(PHICheckerResult)

checker_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CHECKER_SYSTEM),
        (
            "human",
            "Identify if the following question contains PHI or is irrelevant to anything medical or health-related \n\n{question}"
        )
    ]
)

checker = checker_prompt | checker_llm

# SETTING UP RETRIEVERS

retriever = LIGraphRetriever()
pubmed_retriever = PubMedRetriever(top_k_results=2)


# STATE MACHINE FUNCS

def generate_check(state: dict) -> dict:
    
    question = state['question']
    
    # generate is_invalid
    is_invalid = checker.invoke({"question": question}).is_invalid

    return {
        "question": question,
        "is_invalid": is_invalid
    }
    

def question_check(state: dict) -> dict:
    """Checks for PHI or irrelevancy in input question

    Args:
        `state` (`dict`): current graph state
    
    """
    
    # check
    is_invalid = state['is_invalid']
    
    if is_invalid.lower() == "no":
        return 'retrieve'
    else:
        return 'end'
    

def retrieve(state: dict) -> dict:
    """Retrieve chains from the knowledge graph

    Args:
        `state` (`dict`): current graph state
    """
    question = state['question']

    # retrieval
    documents = retriever.get_relevant_chains(question)
    return {
        "documents": documents,
        "question": question
    }
    

def generate(state: dict) -> dict:
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    
    question = state['question']
    documents = state['documents']
    
    generation = rag_generator.invoke({
        "context": documents, "question": question
    })

    return {
        "documents": documents,
        "question": question,
        "generation": generation
    }


def grade_documents(state: dict) -> dict:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """
    
    question = state['question']
    chains = state['documents']
    
    # score each doc
    filtered_docs = []
    web_search = "NO"
    for chain in chains:
        score = grader.invoke({
            "question": question, "chain": chain
        })
        
        grade = score.binary_score
        if grade.lower() == "yes":
            filtered_docs.append(chain)
        else:
            web_search = "YES"
            continue
    
    return {
        "documents": filtered_docs,
        "question": question,
        "web_search": web_search
    }
    
    
def transform_query(state: dict) -> dict:
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """
    
    question = state['question']
    documents = state['documents']
    
    # rewrite question into entities for search
    entities = rewriter.invoke({
        "question": question
    })
    
    return {
        "documents": documents,
        "question": entities
    }
    
    
def web_search(state):
    """
    Web search based on the re-phrased question through pubmed archives

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """
    
    question = state['question']
    documents = state['documents']
    
    # pubmed search
    print(question)
    pubmed_docs = pubmed_retriever.invoke(question)
    pubmed_docs = "\n".join(
        [d.page_content for d in pubmed_docs]
    )
    
    documents.append(pubmed_docs)
    
    return {
        "documents": documents,
        "question": question
    }


def decide_to_generate(state: dict) -> dict:
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    question = state['question']
    web_search = state['web_search']
    filtered_documents = state['documents']
    
    if web_search == "YES":
        return 'transform_query'
    else:
        return 'generate'


# # LANGGRAPH SETUP
# def setup_state_machine():
#     workflow = StateGraph(GraphState)

#     # Define the nodes
#     workflow.add_node("generate_check", generate_check)
#     workflow.add_node("retrieve", retrieve)  # retrieve
#     workflow.add_node("grade_documents", grade_documents)  # grade documents
#     workflow.add_node("generate", generate)  # generatae
#     workflow.add_node("transform_query", transform_query)  # transform_query
#     workflow.add_node("web_search_node", web_search)  # web search

#     # Build graph
#     workflow.set_entry_point("generate_check")
#     # workflow.add_edge("generate_check", "question_check")
#     workflow.add_conditional_edges(
#         "generate_check",
#         question_check,
#         {
#             "retrieve": "retrieve",
#             "end": END
#         }
#     )
#     workflow.add_edge("retrieve", "grade_documents")
#     workflow.add_conditional_edges(
#         "grade_documents",
#         decide_to_generate,
#         {
#             "transform_query": "transform_query",
#             "generate": "generate",
#         },
#     )
#     workflow.add_edge("transform_query", "web_search_node")
#     workflow.add_edge("web_search_node", "generate")
#     workflow.add_edge("generate", END)

#     # Compile
#     runnable = workflow.compile()
    
#     return runnable


# def generate_response(user_prompt: str, runnable) -> str:
#     input = {"question": user_prompt}

#     # Run
#     for output in runnable.stream(input):
#         for key, value in output.items():
#             # Node
#             print(f"Node '{key}':")
#         print("\n---\n")

#     # Final generation
#     try:
#         res = value["generation"]
#         return res
#     except KeyError:
#         return "Whoops, this is an invalid question. Please try again."