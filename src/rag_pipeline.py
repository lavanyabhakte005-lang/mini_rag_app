from src.retriever import retrieve_documents, rerank_documents
from src.llm import generate_answer


def run_rag(question):
    # Step 1: Retrieve Top 10 from Qdrant
    documents = retrieve_documents(question, top_k=10)

    print("\n==============================")
    print("TOP 10 RETRIEVED DOCUMENTS")
    print("==============================")

    for i, document in enumerate(documents, start=1):
        print(f"\nResult {i}")
        print("Similarity Score:", document.score)
        print("Page:", document.payload.get("page"))

    # Step 2: Rerank and select Top 3
    top_3 = rerank_documents(
        question,
        documents,
        top_k=3
    )

    print("\n==============================")
    print("TOP 3 AFTER RERANKING")
    print("==============================")

    for i, item in enumerate(top_3, start=1):
        document = item["document"]

        print(f"\nResult {i}")
        print("Rerank Score:", item["score"])
        print("Page:", document.payload.get("page"))
        print("Text:", document.payload.get("text")[:300])

    # Step 3: Send Top 3 context to Nugen
    answer = generate_answer(question, top_3)

    return answer


if __name__ == "__main__":

    question = "What are the major tourist attractions in India?"

    print("\nQuestion:", question)

    answer = run_rag(question)

    print("\n==============================")
    print("FINAL ANSWER")
    print("==============================")
    print(answer)