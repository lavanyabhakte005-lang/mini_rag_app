from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient


COLLECTION_NAME = "tourism_documents"

# Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Reranker model
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Qdrant
client = QdrantClient(path="qdrant_storage")


def retrieve_documents(question, top_k=10):

    question_embedding = embedding_model.encode(question).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=question_embedding,
        limit=top_k,
        with_payload=True
    )

    return results.points


def rerank_documents(question, documents, top_k=3):

    pairs = []

    for document in documents:
        text = document.payload.get("text", "")
        pairs.append((question, text))

    scores = reranker.predict(pairs)

    ranked = []

    for document, score in zip(documents, scores):
        ranked.append({
            "document": document,
            "score": float(score)
        })

    ranked.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return ranked[:top_k]


if __name__ == "__main__":

    question = "What are the major tourist attractions in India?"

    # -------------------------
    # Top 10 Retrieval
    # -------------------------

    documents = retrieve_documents(question, top_k=10)

    print("\n================================")
    print("TOP 10 BEFORE RERANKING")
    print("================================")

    for i, document in enumerate(documents, start=1):

        print(f"\n{i}. Similarity Score: {document.score}")
        print(f"Page: {document.payload.get('page')}")
        print(f"Text: {document.payload.get('text')[:300]}")


    # -------------------------
    # Reranking
    # -------------------------

    top_3 = rerank_documents(
        question,
        documents,
        top_k=3
    )

    print("\n================================")
    print("TOP 3 AFTER RERANKING")
    print("================================")

    for i, item in enumerate(top_3, start=1):

        document = item["document"]

        print(f"\n{i}. Rerank Score: {item['score']}")
        print(f"Page: {document.payload.get('page')}")
        print(f"Text: {document.payload.get('text')[:500]}")