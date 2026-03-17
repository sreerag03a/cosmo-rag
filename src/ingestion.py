import pymupdf

def extract_pdf(path):
    preprint = pymupdf.open(path)
    text=""
    for page in preprint:
        p_text = page.get_text()
        text+=p_text
    text = text.replace("\n"," ")
    return text

    

if __name__ == "__main__":
    doc_path="./data/2405.06750v2.pdf"

    extracted_text = extract_pdf(doc_path)

    print(len(extracted_text))