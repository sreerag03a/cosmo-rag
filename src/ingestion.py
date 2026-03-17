import pymupdf
from chunking import chunk_creator
import os

def extract_pdf(path):
    '''
    Takes pdf and extracts text from the pdf.
    '''
    preprint = pymupdf.open(path)
    text=""
    for page in preprint:
        p_text = page.get_text()
        text+=p_text
    text = text.replace("\n"," ")
    return text

def extract_from_pdfs(paths,chunk_size=400,chunk_overlap=100):
    document = []
    for path in paths:
        pdf_text = extract_pdf(path)
        p_chunks = chunk_creator(pdf_text,chunk_size=chunk_size,chunk_overlap=chunk_overlap)
        for chunk in p_chunks:
            document.append({
                'text' : chunk,
                'src' : os.path.basename(path)
            })
    return document


if __name__ == "__main__":
    doc_path="./data/2405.06750v2.pdf"

    extracted_text = extract_pdf(doc_path)

    print(len(extracted_text))