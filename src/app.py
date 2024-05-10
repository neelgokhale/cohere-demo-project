import os

from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

# from langsmith.run_helpers import get_current_run_tree
# from langsmith import traceable

from langchain.callbacks import tracing_v2_enabled
from langsmith import Client

from state import *
import streamlit as st

import warnings
warnings.filterwarnings('ignore')

PROJECT_ID = "7dfda498-dc69-4fa7-b932-9f05dd74dd24"


# LANGGRAPH SETUP

def setup_state_machine():
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("generate_check", generate_check)
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generatae
    workflow.add_node("transform_query", transform_query)  # transform_query
    workflow.add_node("web_search_node", web_search)  # web search

    # Build graph
    workflow.set_entry_point("generate_check")
    # workflow.add_edge("generate_check", "question_check")
    workflow.add_conditional_edges(
        "generate_check",
        question_check,
        {
            "retrieve": "retrieve",
            "end": END
        }
    )
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "web_search_node")
    workflow.add_edge("web_search_node", "generate")
    workflow.add_edge("generate", END)

    # Compile
    runnable = workflow.compile()
    
    return runnable


def generate_response(user_prompt: str, runnable) -> str:
    input = {"question": user_prompt}

    # Run
    with tracing_v2_enabled() as cb:
        res = runnable.invoke(input)
        url = cb.get_run_url()
        
    print(url)
    
    # Final generation
    if res['is_invalid'] == "no":     
        return res['generation'], False, url
    else:
        return \
            "Whoops, this is an invalid question. Please try again.", True, url
        
def main():
    
    # SESSION STATE
    if "generation" not in st.session_state:
        st.session_state['generation'] = ""
    if "url" not in st.session_state:
        st.session_state['url'] = ""
    if "is_invalid" not in st.session_state:
        st.session_state['is_invalid'] = True
    if "runnable" not in st.session_state:
        st.session_state['runnable'] = setup_state_machine()
    if "link_disabled" not in st.session_state:
        st.session_state['link_disabled'] = True

    disabled = st.session_state['link_disabled']
    generation = st.session_state['generation']
    
    st.title("HealthCentral: Medical Assistant")
    st.divider()
    
    # input
    with st.container(border=True):
        with st.form("input_form"):
            st.subheader("Input Question")
            user_prompt = st.text_area(
                label="",
                placeholder="Enter your question here.",
                label_visibility="collapsed"
            )
        
            generate = st.form_submit_button("Generate", use_container_width=True)
            if user_prompt and generate:
                with st.spinner("Generating answer..."):
                    generation, is_invalid, url = generate_response(user_prompt=user_prompt,
                                                   runnable=st.session_state['runnable'])
                
                st.session_state['generation'] = generation
                st.session_state['is_invalid'] = is_invalid
                st.session_state['url'] = url
                st.session_state['link_disabled'] = False
                
                if is_invalid:
                    st.warning("Assistant has safeguarded the response to avoid any potential data leakage issues.")
                else:
                    st.success("Response successfully generated.")
            
    # output
    with st.container(border=True):
        st.subheader("Generated Answer")
        st.markdown(st.session_state['generation'])
        
        # metadata
        st.divider()
    
        with st.expander("Trace Metadata"):
            st.link_button("GO TO TRACE", url=st.session_state['url'], disabled=st.session_state['link_disabled'], use_container_width=True)
        

if __name__ == "__main__":
    main()
