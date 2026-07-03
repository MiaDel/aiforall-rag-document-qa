from retrievers.retriever import retrieve
from llm.llm import ask_llm


def ask(question):

    result = retrieve(question)

    docs = result["documents"][0]

    context = "\n\n".join(docs)

    answer = ask_llm(
        context,
        question
    )

    return answer