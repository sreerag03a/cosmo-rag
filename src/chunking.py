from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingestion import extract_pdf
import os

def chunk_creator(text):
    chunker = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
    chunks = chunker.split_text(text)
    return chunks


if __name__ == "__main__":
    doc_path=os.path.join(os.getcwd(),'data/2405.06750v2.pdf')

    extracted_text = extract_pdf(doc_path)
    chunks = chunk_creator(extracted_text)
    print(len(chunks))
    print(chunks[50])