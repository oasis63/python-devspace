from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import faiss

# Load pre-trained model tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load pre-trained model
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get embeddings for a list of texts
def get_embeddings(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()  # Use the [CLS] token embedding

# Example texts
texts = ["Hello, how are you?", "I am doing well.", "What is your name?", "where are you ?", "what are you doing ?", "what is your favourite food?"]

# Get embeddings
embeddings = get_embeddings(texts)

# Create a FAISS index
d = embeddings.shape[1]  # dimension of the embeddings
index = faiss.IndexFlatL2(d)
index.add(embeddings)

# Query the index
# query_text = "How are you?"
query_text = "what is your favourite food?"

query_embedding = get_embeddings([query_text])
D, I = index.search(query_embedding, k=2)

print("Query text:", query_text)
print("Nearest neighbor indices:", I)
print("Nearest neighbor distances:", D)