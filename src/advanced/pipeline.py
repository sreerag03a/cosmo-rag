import numpy as np
import os
import pickle
import faiss
from dotenv import load_dotenv
load_dotenv()


from advanced.ingestion import extract_from_pdfs
from advanced.embed import embed_index,retrieve,detect_query_type,load_model
from advanced.agent import score_chunk, get_llm


def load_or_buildIndex():
    cachepath = os.path.join(os.getcwd(),'cached')
    if os.path.exists(os.path.join(os.getcwd(),'cached','chunks.pkl')):
        print('Loading from cache...')
        index = faiss.read_index(os.path.join(cachepath,'faiss.index'))
        with open(os.path.join(cachepath,'chunks.pkl'),'rb') as f:
            doc_chunks = pickle.load(f)
        conv = []
        for doc in doc_chunks:
            for t in doc['text']:
                conv.append((t,doc['section'],doc['source']))
        sections = [doc['section'] for doc in doc_chunks]
    else:
        print('Building index from scratch...')
        doc_paths=[os.path.join(os.getcwd(),'data','WSQ.pdf')]
        doc_chunks = extract_from_pdfs(doc_paths,chunk_size=400,chunk_overlap=150)
        conv = []
        for doc in doc_chunks:
            for t in doc['text']:
                conv.append((t,doc['section'],doc['source']))
        sections = [doc['section'] for doc in doc_chunks]
        model,index = embed_index(doc_chunks)
        faiss.write_index(index,os.path.join(cachepath,'faiss.index'))
        with open(os.path.join(cachepath,'chunks.pkl'),'wb') as f:
            pickle.dump(doc_chunks,f)
    return index,sections,conv


def rag_pipeline(query,model,index,sections,conv,on_device=False,model_choice = "llama-3.1-8b-instant"):
    indices = retrieve(query,model,index,n_res=10)
    allowed_sections = detect_query_type(query,sections)

    retrieved = []
    filtered = []
    for i,item in enumerate(conv):
        if i in indices:
            retrieved.append(item)
    scored = [(doc,score_chunk(doc[0],query)) for doc in retrieved]
    scored.sort(key=lambda x:x[1],reverse=True)
    retrieved = [doc for doc,score in scored[:3]]
    filtered = []
    for t,section,src in retrieved:
        for section_ in allowed_sections:
            if section in section_:
                filtered.append((t,section,src))
    if not filtered:
        filtered = retrieved
    print(len(filtered))
    prompt = f"Use only the provided cosmology context to answer the query : {query}\n\nContext:\n"
    for t in filtered:
        prompt += f"Source :{t[2]}, text : {t[0]}"
    prompt += "\nProvide a clear and brief answer.\nDo NOT add information not explicitly mentioned.\nDo NOT assume common cosmological datasets."
    llm = get_llm(on_device=on_device,model_choice=model_choice)
    # if on_device == False:
    #     from langchain_groq import ChatGroq
    #     llm = ChatGroq(
    #         model = model_choice,
    #         temperature=0.2,
    #         api_key=os.getenv("GROQ_KEY")
    #     )
        
    # else:
    #     from langchain_ollama import ChatOllama
    #     llm = ChatOllama(
    #         model=model_choice,
    #         temperature=0.2
    #     )
    response = llm.invoke(prompt)
    reply = response.content if hasattr(response, "content") else str(response)
    print('RAG Response:')
    print(reply)
    return reply,filtered




if __name__ =='__main__':
    # doc_paths=[os.path.join(os.getcwd(),'data','WSQ.pdf')]
    # doc_chunks = extract_from_pdfs(doc_paths,chunk_size=300,chunk_overlap=150)
    # conv = []
    # for doc in doc_chunks:
    #     for t in doc['text']:
    #         conv.append((t,doc['section'],doc['source']))
    # sections = [doc['section'] for doc in doc_chunks]
    # # print(sections)
    # model,index = embed_index(doc_chunks)
    # start = time()
    index,sections,conv = load_or_buildIndex()
    model = load_model()
    # end = time()
    # print(f'Time taken : {end-start} seconds')
    query = input("Ask a question : ")
    rag_pipeline(query,model,index,sections,conv, on_device=True, model_choice="gemma:2b")

    