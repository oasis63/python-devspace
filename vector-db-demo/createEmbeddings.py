from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained model tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize input text
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors='pt')

# print("tokens : inputs ", inputs)

# Load pre-trained model
model = BertModel.from_pretrained('bert-base-uncased')

# Get the embeddings
with torch.no_grad():
    outputs = model(**inputs)

# The embeddings are in the last hidden state
embeddings = outputs.last_hidden_state

print(embeddings)