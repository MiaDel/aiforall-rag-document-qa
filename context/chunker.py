from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

def create_chunks(docs):
    chunks = []

    for doc in docs:
        pieces = splitter.split_text(doc["text"])

        for piece in pieces:
            chunks.append({
                "text": piece,
                "page": doc["page"],
                "source": doc["source"]
            })

    return chunks