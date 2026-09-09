import numpy as np


def cosine_search(query_embedding, review_embeddings, k):
    query_embedding = query_embedding.reshape(-1)
    dot_products = np.dot(review_embeddings, query_embedding)

    query_norm = np.linalg.norm(query_embedding)
    review_norms = np.linalg.norm(review_embeddings, axis=1)

    cosine_similarities = dot_products / (query_norm * review_norms)
    top_indices = np.argsort(-cosine_similarities)[:k]
    top_cosine_similarities = cosine_similarities[top_indices]

    return top_indices, top_cosine_similarities
