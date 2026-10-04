import fitz
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


# -----------------------------
# Settings
# -----------------------------

PDF_PATH = "documents/TOURISM IN INDIA.pdf"
COLLECTION_NAME = "tourism_documents"

CHUNK_SIZE = 500
OVERLAP = 100


# -----------------------------
# Load embedding model
# -----------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# Connect to Qdrant
# -----------------------------

client = QdrantClient(path="qdrant_storage")


# -----------------------------
# Create collection
# -----------------------------

if not client.collection_exists(COLLECTION_NAME):

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )


# -----------------------------
# Extract and create chunks
# -----------------------------

doc = fitz.open(PDF_PATH)

points = []
point_id = 1


for page_number, page in enumerate(doc, start=1):

    text = page.get_text()

    start = 0

    while start < len(text):

        end = start + CHUNK_SIZE

        chunk = text[start:end].strip()

        if chunk:

            # Create embedding
            embedding = model.encode(chunk).tolist()

            # Store chunk + metadata
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "text": chunk,
                    "page": page_number,
                    "source": PDF_PATH
                }
            )

            points.append(point)

            point_id += 1

        start += CHUNK_SIZE - OVERLAP


doc.close()


# -----------------------------
# Insert into Qdrant
# -----------------------------

client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)


print("================================")
print("PDF data stored in Qdrant!")
print("================================")
print("Total chunks:", len(points))
print("Collection:", COLLECTION_NAME)