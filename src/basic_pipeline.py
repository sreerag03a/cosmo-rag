import numpy as np
import os
from ingestion import extract_pdf
from chunking import chunk_creator
from embed import embed_index,retrieve




if __name__ == "__main__":
    doc_path=os.path.join(os.getcwd(),'data/2405.06750v2.pdf')

    extracted_text = extract_pdf(doc_path)
    chunks = chunk_creator(extracted_text,chunk_size=300,chunk_overlap=100)

    model,index = embed_index(chunks)
    query = input("Ask question : ")
    indices = retrieve(query,model,index,n_res=3)
    
    context =  "\n\n".join([chunks[i] for i in indices])
    prompt = f"""
Answer ONLY using the context below.
If the answer is not in the context, say "Not found".

Context:
{context}

Question:
{query}
"""

import ollama

response = ollama.chat(
    model="phi3:mini",
    messages=[{"role": "user", "content": prompt}]
)

reply = response['message']['content']
print("\nAnswer :\n")
print(reply)
