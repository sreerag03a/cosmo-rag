import streamlit as st
import os

from advanced.ingestion import extract_from_pdfs
from advanced.embed import embed_index
from advanced.pipeline import rag_pipeline

st.title("CosmoRAG")
doc_paths = []
mode = st.radio(
    "Choose input mode:",
    ["Use default PDFs", "Upload PDFs"]
)
if mode=="Use default PDFs":
    items = [os.path.join(os.getcwd(),'data','WSQ.pdf')]
    for item in items:
        doc_paths.append(item)
    # print(doc_paths)
    doc_chunks = extract_from_pdfs(doc_paths)
    docs_formatted = []
    for doc in doc_chunks:
        for t in doc['text']:
            docs_formatted.append((t,doc['section'],doc['source']))
    sections = [doc['section'] for doc in doc_chunks]
    model,index = embed_index(doc_chunks)





else:
    uploaded_files = st.file_uploader(
        "Upload PDF(s)",
        type="pdf",
        accept_multiple_files=True
    )


    if uploaded_files:
        delfiles = []
        for file in uploaded_files:
            datapath = os.path.join(os.getcwd(),"data")
            os.makedirs(datapath, exist_ok=True)
            path = os.path.join(datapath,file.name)
            delfiles.append(path)
            with open(path,"wb") as f:
                f.write(file.read())
            doc_paths.append(path)

        items = [os.path.join(os.getcwd(),'data','WSQ.pdf')]
        for item in items:
            doc_paths.append(item)
        # print(doc_paths)
        doc_chunks = extract_from_pdfs(doc_paths)
        for file in delfiles:
            os.remove(file)
        docs_formatted = []
        for doc in doc_chunks:
            for t in doc['text']:
                docs_formatted.append((t,doc['section'],doc['source']))
        sections = [doc['section'] for doc in doc_chunks]
        model,index = embed_index(doc_chunks)



        
model_choice = st.selectbox(
        "Select model",
        ["phi3:mini", "gemma:2b"]
    )
if "history" in st.session_state:
    for item in st.session_state.history:
                st.write("**You:**", item["query"])
                st.write("**AI:**", item["answer"])
query = st.text_input("Ask a question: ")
if query:
    if "history" not in st.session_state:
        st.session_state.history = []
    with st.spinner("Thinking..."):
        answer = rag_pipeline(query,model,index,sections,docs_formatted,model_choice)
        st.session_state.history.append({
    "query": query,
    "answer": answer
})
        
        st.markdown("### Answer")
        st.write(answer)
