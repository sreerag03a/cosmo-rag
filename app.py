import streamlit as st
import os

from advanced.ingestion import extract_from_pdfs
from advanced.embed import embed_index
from advanced.pipeline import rag_pipeline

doc_paths=[os.path.join(os.getcwd(),'data','WSQ.pdf'),os.path.join(os.getcwd(),'data','LDEM.pdf'),os.path.join(os.getcwd(),'data','PDDE.pdf')]
doc_chunks = extract_from_pdfs(doc_paths)
docs_formatted = []
for doc in doc_chunks:
    for t in doc['text']:
        docs_formatted.append((t,doc['section'],doc['source']))
sections = [doc['section'] for doc in doc_chunks]
model,index = embed_index(doc_chunks)


st.title("CosmoRAG")

query = st.text_input("Ask a question: ")


if query:
    answer = rag_pipeline(query,model,index,sections,docs_formatted)
    st.write(answer)