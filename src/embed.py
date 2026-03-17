from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from ingestion import extract_pdf,extract_from_pdfs
from chunking import chunk_creator





# def embed_index(chunks):
#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     embeddings = model.encode(chunks)
#     dim = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dim)
#     index.add(np.array(embeddings))
#     return model,index



def embed_index(docs):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    chunks = [doc["text"] for doc in docs]
    embeddings = model.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return model,index

def retrieve(query,model,index,n_res = 3):
    q_embed = model.encode([query])
    distances,indices = index.search(q_embed,n_res)
    return indices[0]


if __name__ == "__main__":
    doc_paths=[os.path.join(os.getcwd(),'data','2405.06750v2.pdf'),os.path.join(os.getcwd(),'data','2406.18095v1.pdf'),os.path.join(os.getcwd(),'data','2308.03084v1.pdf')]

    extracted_text = extract_from_pdfs(doc_paths)

    model,index = embed_index(extracted_text)
    query = "What is the scalar field potential?"
    indices = retrieve(query,model,index)

    for i in indices:
        print(extracted_text[i])
        print('------')
