from advanced.pipeline import rag_pipeline
from advanced.ingestion import extract_from_pdfs
from advanced.embed import embed_index
import os
import numpy as np
from itertools import product

datapath = os.path.join(os.getcwd(),"data","WSQ.pdf")


def hitrate(retrieved,keywords):
    a = 0
    for doc in retrieved:
        text = doc[0].lower()
        if any (k.lower() in text for k in keywords):
            a+=1
    return a/len(keywords)


def groundedness(answer,retrieved):
    context = " ".join(doc[0] for doc in retrieved).lower()
    answer_w = answer.lower().split()
    overlap_w = sum(1 for w in answer_w if w in context)

    return overlap_w/len(answer_w)


if __name__ == '__main__':
    # doc_paths=[os.path.join(os.getcwd(),'data','WSQ.pdf')]
    # doc_chunks = extract_from_pdfs(doc_paths)
    # conv = []
    # for doc in doc_chunks:
    #     for t in doc['text']:
    #         conv.append((t,doc['section'],doc['source']))
    # sections = [doc['section'] for doc in doc_chunks]
    # # print(sections)
    # model,index = embed_index(doc_chunks)
    # query = "Which dataset is used in this work?"
    # keywords = ["SNIa", "OHD", "Pantheon+", "DES", "Dark Energy Survey"]
    # answer,retrieved = rag_pipeline(query,model,index,sections,conv,model_choice="phi3:mini")
    # print(f'Hitrate : {hitrate(retrieved,keywords)}')
    # print(f'Groundedness : {groundedness(answer,retrieved)}')

    queries = ["Which dataset is used in this work?", "Which potential is used in this work?", "What is the name given to the model in this work?"]
    keywords = [["SNIa", "OHD", "Pantheon+", "DES", "Dark Energy Survey"],["Woods-Saxon", "nuclear", "physics"],["WSQ", "Woods-Saxon Quintessence"]]
    hitrates = []
    gr = []
    # for _ in range(10):
    #     for q,k in zip(queries,keywords):
    #         answer,retrieved = rag_pipeline(q,model,index,sections,conv,model_choice="qwen3-vl:4b")
    #         hitrates.append(hitrate(retrieved,k))
    #         groundedness_.append(groundedness(answer,retrieved))
    # print(f'Hitrates (10 pass avg) : {np.mean(hitrates)}')
    # print(f'Groundedness (10 pass avg): {np.mean(groundedness_)}')

    chunk_sizes = [300,400,500,800,1000,1200]
    chunk_overlaps = [50,80,100,150,200]
    combs = list(product(chunk_sizes,chunk_overlaps))

    doc_paths=[os.path.join(os.getcwd(),'data','WSQ.pdf')]

    for size,overlap in combs:
        doc_chunks = extract_from_pdfs(doc_paths,chunk_size=size,chunk_overlap=overlap)
        conv = []
        for doc in doc_chunks:
            for t in doc['text']:
                conv.append((t,doc['section'],doc['source']))
        sections = [doc['section'] for doc in doc_chunks]
        # print(sections)
        model,index = embed_index(doc_chunks)
        groundedness_ = []
        curr_hitrates = []
        for _ in range(2):
            for q,k in zip(queries,keywords):
                answer,retrieved = rag_pipeline(q,model,index,sections,conv,model_choice="phi3:mini")
                curr_hitrates.append(hitrate(retrieved,k))
                groundedness_.append(groundedness(answer,retrieved))
        hitrates.append(np.mean(curr_hitrates))
        gr.append(np.mean(groundedness_))
    
    sortedlist = list(zip(hitrates,gr,combs))
    print(sortedlist)
    sortedlist.sort(key=lambda x: ((2*x[0]*x[1])/(x[0]+x[1])), reverse=True)
    print(f'Best chunk size : {sortedlist[0][2][0]}. Best chunk overlap : {sortedlist[0][2][1]}')
    
