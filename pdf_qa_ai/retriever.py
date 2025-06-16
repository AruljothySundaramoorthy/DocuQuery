import numpy as np
import faiss
from sklearn.metrics.pairwise import cosine_similarity

def build_index(embeddings):
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index

def find_top_chunks(question_embedding, chunk_embeddings, chunks, top_k=3):
    similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices]
