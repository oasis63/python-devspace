from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from sentence_transformers import SentenceTransformer
import numpy as np

# Step 3: Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Step 4: Define Collection Schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
]

schema = CollectionSchema(fields, "Text embeddings collection")
collection = Collection("text_embeddings", schema)

# Step 5: Load a Pre-trained Model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 6: Generate Embeddings and Insert Data
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Milvus is an open-source vector database.",
    "Python is a popular programming language.",
    "Machine learning is a subset of artificial intelligence."
]

embeddings = model.encode(texts)

entities = [
    texts,
    embeddings.tolist()
]

collection.insert(entities)

# Step 7: Create an Index
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128}
}

collection.create_index("embedding", index_params)

# Step 8: Perform a Similarity Search
query = "What is Milvus?"
query_embedding = model.encode([query])[0].tolist()

search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 10}
}

results = collection.search([query_embedding], "embedding", search_params, limit=2, output_fields=["text"])

for result in results:
    for item in result:
        print(f"Distance: {item.distance}, Text: {item.entity.get('text')}")

# Step 9: Disconnect from Milvus
connections.disconnect("default")