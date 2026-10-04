from sentence_transformers import SentenceTransformer

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Test text
text = "Tourism in India includes many historical and cultural destinations."

# Create embedding
embedding = model.encode(text)

print("Embedding created successfully!")
print("Embedding size:", len(embedding))
print("First 5 values:", embedding[:5])