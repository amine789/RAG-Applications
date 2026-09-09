import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def cosine_search(query_embedding, review_embeddings, k):
    query_embedding = query_embedding.reshape(-1)
    dot_products = np.dot(review_embeddings, query_embedding)

    query_norm = np.linalg.norm(query_embedding)
    review_norms = np.linalg.norm(review_embeddings, axis=1)

    cosine_similarities = dot_products / (query_norm * review_norms)
    top_indices = np.argsort(-cosine_similarities)[:k]
    top_cosine_similarities = cosine_similarities[top_indices]

    return top_indices, top_cosine_similarities


def tfidf_search(query, vectorizer, tfidf_matrix, documents, k=3):
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, tfidf_matrix)[0]
    top_indices = scores.argsort()[::-1][:k]
    return [(documents[i], scores[i]) for i in top_indices]


def bm25_search(query, bm25_index, documents, k=3):
    tokenized_query = query.lower().split()
    scores = bm25_index.get_scores(tokenized_query)
    top_indices = scores.argsort()[::-1][:k]
    return [(documents[i], scores[i]) for i in top_indices]
