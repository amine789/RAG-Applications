from sklearn.metrics.pairwise import cosine_similarity


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
