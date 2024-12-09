import numpy as np
import faiss

# Generate some example embeddings
d = 64  # dimension of the embeddings
nb = 1000  # number of vectors
np.random.seed(1234)
xb = np.random.random((nb, d)).astype('float32')

# Create a FAISS index
index = faiss.IndexFlatL2(d)  # L2 distance index
index.add(xb)  # Add vectors to the index

# Query the index
nq = 10  # number of queries
xq = np.random.random((nq, d)).astype('float32')
D, I = index.search(xq, k=5)  # Search for the 5 nearest neighbors

print("Indices of nearest neighbors:", I)
print("Distances to nearest neighbors:", D)