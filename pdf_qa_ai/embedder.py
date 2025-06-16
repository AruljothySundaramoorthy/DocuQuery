from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_chunks(chunks):
    return model.encode(chunks)

def embed_question(question):
    return model.encode([question])
