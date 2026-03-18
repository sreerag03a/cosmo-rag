import pymupdf
from baseline.chunking import chunk_creator
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