from langchain_text_splitters import RecursiveCharacterTextSplitter
# from ingestion import extract_pdf
import os

def chunk_creator(text,chunk_size=500,chunk_overlap=100):
    '''
    Divides text into chunks of size 500 with 100 characters overlap
    '''
    chunker = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    chunks = chunker.split_text(text)
    return chunks


# if __name__ == "__main__":
    # doc_paths=os.path.join(os.getcwd(),'data','2405.06750v2.pdf')

    # extracted_text = extract_pdf(doc_paths)
    # print(extracted_text)
    # chunks = chunk_gen(extracted_text)
    # print(len(chunks))
    # print(chunks[51])