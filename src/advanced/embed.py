from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from advanced.ingestion import extract_from_pdfs
from advanced.chunking import chunk_per_doc_sec

def embed_index(docs):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = []
    for doc in docs:
        for t in doc['text']:
            texts.append(t)

    embeddings = model.encode(texts)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return model,index


def retrieve(query,model,index,n_res = 3):
    q_embed = model.encode([query])
    distances,indices = index.search(q_embed,n_res)
    return indices[0]

def detect_query_type(query,sections):
    q = query.lower()

    if "model" in q:
        return ["introduction", "model"]
    elif "data" in q or "analysis" in q:
        return ["method", "analysis", "introduction"]
    elif "result" in q:
        return ["results", "data", "analysis"]
    else:
        return sections
