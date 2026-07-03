from context.retrievers.retriever import retrieve
from llm.llm import ask_llm


def ask(question):

    result = retrieve(question)

    docs = result["documents"][0]
    metas = result["metadatas"][0]

    context = "\n\n".join(docs)

    answer = ask_llm(
        context=context,
        question=question
    )

    citations = []

    for meta in metas:
        citations.append(
            f"{meta['source']} | Page {meta['page']}"
        )

    return answer, citations