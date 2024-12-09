import cohere
import os
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

load_dotenv()

# connection to milvus db

# connections.connect("default", host="192.168.0.103", port="19530")


# Retrieve the API key from the environment variable
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Initialize the Cohere client with your API key
co = cohere.Client(COHERE_API_KEY)

# Define the texts you want to embed
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Natural language processing is a fascinating field.",
]

# Generate embeddings for the texts
response = co.embed(texts=texts)
embeddings = response.embeddings

# Print the embedding for the first sentence
print("Embedding for Sentence 1:", embeddings[0])

# Generate embedding for the query text
query_text = "Artificial intelligence is revolutionizing technology."
query_embedding = co.embed(texts=[query_text]).embeddings[0]

# Convert embeddings to numpy arrays
embeddings_np = np.array(embeddings)
query_embedding_np = np.array(query_embedding).reshape(1, -1)

# Calculate cosine similarity
similarities = cosine_similarity(query_embedding_np, embeddings_np)


print("similarities : ", similarities)

# Print the similarities
for i, text in enumerate(texts):
    print(f"Text: {text}, Similarity: {similarities[0][i]}")

# Get the indices that would sort the similarities in descending order
sorted_indices = np.argsort(similarities[0])[::-1]

# Print the top N most similar texts
top_n = 2
for i in range(top_n):
    index = sorted_indices[i]
    print(f"Rank {i+1}: Text: {texts[index]}, Similarity: {similarities[0][index]}")
