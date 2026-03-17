from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from ingestion import extract_pdf
from chunking import chunk_creator





def embed_index(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return model,index



if __name__ == "__main__":
    doc_path=os.path.join(os.getcwd(),'data/2405.06750v2.pdf')

    extracted_text = extract_pdf(doc_path)
    chunks = chunk_creator(extracted_text,chunk_size=300,chunk_overlap=80)

    model,index = embed_index(chunks)
    query = "What is the scalar field potential?"
    query_embed = model.encode([query])

    k = 4
    distances,indices = index.search(query_embed,k)
    print('Results : \n')

    for i in indices[0]:
        print(chunks[i])
        print('------')
