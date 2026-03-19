import numpy as np
import os
from advanced.ingestion import extract_from_pdfs
from advanced.chunking import chunk_per_doc_sec
from advanced.embed import embed_index,retrieve,detect_query_type
from baseline.basic_pipeline import score_chunk
import ollama



if __name__ =='__main__':
    doc_paths=[os.path.join(os.getcwd(),'data','WSQ.pdf'),os.path.join(os.getcwd(),'data','LDEM.pdf'),os.path.join(os.getcwd(),'data','PDDE.pdf')]
    doc_chunks = extract_from_pdfs(doc_paths)
    conv = []
    for doc in doc_chunks:
        for t in doc['text']:
            conv.append((t,doc['section'],doc['source']))
    sections = [doc['section'] for doc in doc_chunks]
    # print(sections)
    model,index = embed_index(doc_chunks)

    query = input("Ask a question : ")
    indices = retrieve(query,model,index,n_res=10)
    allowed_sections = detect_query_type(query,sections)

    retrieved = []
    for i,item in enumerate(conv):
        if i in indices:
            retrieved.append(item)
    filtered = []
    for t,section,src in retrieved:
        for section_ in allowed_sections:
            if section in section_:
                filtered.append((t,section,src))

    # print(filtered)
    prompt = f"Use only the provided cosmology contex to answer the query : {query}\n\nContext:\n"
    
    for t in filtered:
        prompt += f"Source :{t[2]}, text : {t[0]}"
    prompt += "\nProvide a clear and brief answer.\nDo NOT add information not explicitly mentioned.\nDo NOT assume common cosmological datasets."
    response = ollama.chat(
        model='phi3:mini',
        messages=[{"role": "user", "content": prompt}]
    )
    print('RAG Response:')
    print(response['message']['content'])