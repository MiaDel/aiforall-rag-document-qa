from context.pdf_parser import parse_pdf
from context.chunker import create_chunks
from context.retrievers.retriever import (
    index_chunks,
    retrieve
)

# PDF path
path = "data/sample_docs/pdf/sample.pdf"

# Parse PDF
docs = parse_pdf(path)

# Create chunks
chunks = create_chunks(docs)

# Index into ChromaDB
index_chunks(chunks)

print("✅ Indexing Complete")

# Test Question
question = "What is the main topic of the document?"

# Retrieve relevant chunks
result = retrieve(question)

documents = result["documents"][0]
metadatas = result["metadatas"][0]
distances = result["distances"][0]

print("\n" + "=" * 60)
print("QUESTION:")
print(question)
print("=" * 60)

print("\nRETRIEVED CHUNKS:\n")

for i, (doc, meta, dist) in enumerate(
        zip(documents, metadatas, distances), start=1):

    print(f"\nChunk {i}")
    print(f"Page      : {meta['page']}")
    print(f"Source    : {meta['source']}")
    print(f"Distance  : {dist:.4f}")
    print("-" * 60)

    preview = doc[:400]

    if len(doc) > 400:
        preview += "..."

    print(preview)