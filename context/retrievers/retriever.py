import chromadb
from context.embeddings import embed

client = chromadb.PersistentClient(
    path="data/chroma_db"
)

collection = client.get_or_create_collection(
    name="documents"
)


def index_chunks(chunks):
    for i, chunk in enumerate(chunks):

        collection.add(
            ids=[str(i)],
            documents=[chunk["text"]],
            embeddings=[embed(chunk["text"])],
            metadatas=[{
                "page": chunk["page"],
                "source": chunk["source"]
            }]
        )


def retrieve(question):

    result = collection.query(
        query_embeddings=[embed(question)],
        n_results=5
    )

    return result