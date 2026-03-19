from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# def chunk_creator(text,chunk_size=800,chunk_overlap=150):
#     '''
#     Divides text into chunks of size 500 with 100 characters overlap
#     '''
#     chunker = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
#     chunks = chunker.split_text(text)
#     return chunks

def chunk_per_doc_sec(doc_list,chunk_size=800,chunk_overlap=150):
    chunker = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    end_res = []
    for doc in doc_list:
        for sec in doc:
            doc_ = {}
            chunks = chunker.split_text(sec['text'])
            doc_['text'] = chunks
            doc_['source'] = sec['source']
            doc_['section'] = sec['section']
            end_res.append(doc_)
    return end_res
