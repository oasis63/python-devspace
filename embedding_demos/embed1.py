import torch
import requests
from transformers import BertTokenizer, BertModel

# Disable SSL verification
requests.adapters.DEFAULT_RETRIES = 5


# Load pre-trained BERT model and tokenizer from Hugging Face
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")


sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "Natural language processing is a fascinating field.",
]

inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

print("inputs : ", inputs)

with torch.no_grad():
    outputs = model(**inputs)

# check for outputs.last_hidden_state


# Extract the embedding of the [CLS] token (the first token of each sentence)
cls_embeddings = outputs.last_hidden_state[:, 0, :]

# Convert to a NumPy array for easier handling (optional)
sentence_embeddings = cls_embeddings.numpy()

# Extract the embedding of the [CLS] token (the first token of each sentence)
cls_embeddings = outputs.last_hidden_state[:, 0, :]

# Convert to a NumPy array for easier handling (optional)
sentence_embeddings = cls_embeddings.numpy()


print("Sentence Embedding for Sentence 1:", sentence_embeddings[0])
print("Sentence Embedding for Sentence 2:", sentence_embeddings[1])
