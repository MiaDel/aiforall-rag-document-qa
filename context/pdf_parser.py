import fitz

def parse_pdf(file_path):
    docs = []

    pdf = fitz.open(file_path)

    for page_num in range(len(pdf)):
        page = pdf.load_page(page_num)

        docs.append({
            "text": page.get_text(),
            "page": page_num + 1,
            "source": file_path
        })

    return docs