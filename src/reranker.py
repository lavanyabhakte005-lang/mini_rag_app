from sentence_transformers import CrossEncoder

# Reranker model
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank_documents(question, documents, top_k=3):

    pairs = []

    for document in documents:
        text = document.payload.get("text", "")
        pairs.append((question, text))

    scores = reranker.predict(pairs)

    ranked_documents = []

    for document, score in zip(documents, scores):
        ranked_documents.append({
            "document": document,
            "rerank_score": float(score)
        })

    # Highest score first
    ranked_documents.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return ranked_documents[:top_k]