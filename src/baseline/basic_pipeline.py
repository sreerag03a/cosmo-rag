import numpy as np
import os
from baseline.ingestion import extract_pdf,extract_from_pdfs
from baseline.chunking import chunk_creator
from baseline.embed import embed_index,retrieve
import ollama


def score_chunk(chunk, query):
    query_words = set(query.lower().split())
    chunk_words = set(chunk.lower().split())
    return len(query_words & chunk_words)


if __name__ == "__main__":
    doc_paths=[os.path.join(os.getcwd(),'data','WSQ.pdf'),os.path.join(os.getcwd(),'data','LDEM.pdf'),os.path.join(os.getcwd(),'data','PDDE.pdf')]

    extracted_text = extract_from_pdfs(doc_paths,chunk_size=800,chunk_overlap=100)

    model,index = embed_index(extracted_text)
    query = input("Ask question : ")
    indices = retrieve(query,model,index,n_res=5)
    # context = ""
    retrieved = [extracted_text[i] for i in indices]
    scored = [(doc,score_chunk(doc['text'],query)) for doc in retrieved]
    scored.sort(key=lambda x:x[1],reverse=True)
    retrieved = [doc for doc,score in scored[:3]]

    
    print('Filtered : ')
    for doc in retrieved:
        print(doc['text'])
    
    n_prompt = f"Use this context to answer the query : {query}\n\n"

    for doc in retrieved:
        n_prompt += f"{doc['src']}:\n{doc['text']}\n\n"

    n_prompt += "\nProvide a clear and brief answer.\nDo NOT add information not explicitly mentioned.\nDo NOT assume common cosmological datasets."

    response = ollama.chat(
        model='phi3:mini',
        messages=[{"role": "user", "content": n_prompt}]
    )
    print('RAG Response:')
    print(response['message']['content'])
