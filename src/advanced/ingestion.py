import pymupdf
from advanced.chunking import chunk_per_doc_sec
from baseline.ingestion import remove_references
import os
import re


def extract_pdf(path):
    '''
    Takes pdf and extracts text from the pdf.
    '''
    preprint = pymupdf.open(path)
    text=""
    for page in preprint:
        p_text = page.get_text()
        text+=p_text
    # text = text.replace("\n\n"," ")
    # print(len(text))
    return text
def merge_sections(sections):
    merged = {}
    for item in sections:
        key = (item['source'],item['section'])
        if key not in merged:
            merged[key] = {
                'section' : item['section'],
                'text' : item['text'],
                'source' : item['source']
            }
        else:
            merged[key]["text"] += " " + item["text"]
    return list(merged.values())

def normalize_section(section):
    
    if section in ["methods", "methods", "analysis", "data"]:
        return "method"
    elif section in ["result", "results", "conclusion", "discussion"]:
        return "results"
    elif section in ["introduction", "abstract", "model"]:
        return "introduction"
    elif section == "unknown":
        return "unknown"
    else:
        return section
def split_sections(text:str,path):
    section_pattern = re.compile(
        r'^\s*(\d+\.?\s+)?(abstract|introduction|method|methods|data|analysis|results|discussion|conclusion|references)\b',
        re.IGNORECASE
    )
    sections=[]
    current_sec = "unknown"
    current_text = ""
    for line in text.split("\n"):
        line_lower = line.strip().lower()
        is_header=False
        match = section_pattern.match(line_lower)
        if match:
            if len(line_lower) < 60:
                is_header = True
        if match and is_header:
            if current_text.strip():
                sections.append({
            'section': current_sec,
            'text': current_text.strip(),
            'source': os.path.basename(path)
        })
            detected_section = match.group(2).lower()
            current_sec = normalize_section(detected_section)
            current_text = ""
        else:
            current_text += line  + " "
    if current_text.strip():
        sections.append({
            'section': current_sec,
            'text': current_text.strip(),
            'source': os.path.basename(path)
        })
    sections = merge_sections(sections)
    # new_sections= []
    # for sec in sections:
    #     print(sec)
    #     print('_______________________________________')
    #     # if len(sec["text"])>200:
    #     #     new_sections.append(sec)
    # print(fike)
    return sections



def extract_from_pdfs(paths,chunk_size=800,chunk_overlap=150):
    document = []
    for path in paths:
        pdf_text = extract_pdf(path)
        pdf_text = remove_references(pdf_text)
        pdf_in_sec = split_sections(pdf_text,path)
        document.append(pdf_in_sec)
    doc_chunks = chunk_per_doc_sec(document,chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    return doc_chunks


if __name__ =='__main__':
    doc_paths=[os.path.join(os.getcwd(),'data','WSQ.pdf'),os.path.join(os.getcwd(),'data','LDEM.pdf'),os.path.join(os.getcwd(),'data','PDDE.pdf')]
    doc_chunks = extract_from_pdfs(doc_paths)
    print(len(doc_chunks['introduction']))
