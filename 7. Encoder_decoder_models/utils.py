import json
import random
import re
import sys
from uuid import uuid4

import faiss
import numpy as np
import pandas as pd
import torch
from anthropic import Anthropic
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models

sys.path.append("..")
from llm_model import llm

anthropic_client = Anthropic()


# ---- Chunking ----

def recursive_split_documents(data, chunk_size=1024, chunk_overlap=50, separators=None):
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )

    return splitter.create_documents(
        texts=[d["page_content"] for d in data],
        metadatas=[d["metadata"] for d in data],
    )


# ---- Embeddings (raw transformers / nomic-embed-text-v1.5) ----

def get_text_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings[0].detach().numpy()


# ---- Qdrant indexing and retrieval (research-paper pipeline) ----

def index_documents(client, collection_name, documents, embeddings, distance=models.Distance.COSINE):
    vector_size = len(embeddings[0])

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=distance),
        )

    client.upload_points(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=str(uuid4()),
                vector=np.array(embeddings[idx]),
                payload={
                    "metadata": doc.metadata,
                    "content": doc.page_content,
                },
            )
            for idx, doc in enumerate(documents)
        ],
    )


def query_qdrant(query, qdrant_client, tokenizer, model, collection_name="research_collection", limit=5):
    query_em = get_text_embeddings(query, tokenizer, model)
    text_hits = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_em,
        limit=limit,
    ).points

    results = []
    for i, point in enumerate(text_hits):
        results.append({
            "source_id": i + 1,
            "content": point.payload["content"],
            "metadata": point.payload["metadata"],
        })
    return results


# ---- Generation (research-paper pipeline) ----

def generate_answer(query, client, tokenizer, model):
    sources = query_qdrant(query, client, tokenizer, model)

    prompt = f"""
    Based on the following query, generate a comprehensive answer.
    Include citations [1][2] and mention authors, paper titles, and venues.
    Explain concepts clearly.

    Query: "{query}"
    Context: "{sources}"

    Return in Markdown format.
    """

    response = ""
    for chunk in llm.stream(prompt):
        print(chunk.content, end="", flush=True)
        response += chunk.content

    return response, sources


# ---- Evaluation: synthetic eval sets, retrieval, and an LLM judge ----

def build_synthetic_eval_set(documents, n=20, seed=42):
    random.seed(seed)
    sample = random.sample(documents, min(n, len(documents)))

    eval_set = []
    for doc in sample:
        prompt = f"""
        Write exactly one specific question that is fully answered by the passage below.
        Return only the question, nothing else.

        Passage: "{doc.page_content}"
        """
        question = llm.invoke(prompt).content.strip()
        eval_set.append({
            "question": question,
            "source_content": doc.page_content,
            "paper_id": doc.metadata["paper_id"],
        })

    return eval_set


def evaluate_retrieval(eval_set, qdrant_client, tokenizer, model, collection_name="research_collection", k=5):
    hits, reciprocal_ranks = 0, []

    for item in eval_set:
        results = query_qdrant(item["question"], qdrant_client, tokenizer, model, collection_name=collection_name, limit=k)

        rank = None
        for i, r in enumerate(results):
            if r["metadata"]["paper_id"] == item["paper_id"]:
                rank = i + 1
                break

        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)

    return {
        "recall@k": hits / len(eval_set),
        "mrr": sum(reciprocal_ranks) / len(eval_set),
    }


def parse_llm_json(message_content):
    match = re.search(r"[\{\[].*[\}\]]", message_content, re.DOTALL)
    json_str = match.group(0) if match else message_content
    return json.loads(json_str)


def judge_answer(question, context, answer):
    judge_prompt = f"""
    You are evaluating a generated answer against the context it was given.

    Question: "{question}"
    Context: "{context}"
    Answer: "{answer}"

    Score the answer on two dimensions, each from 1 (worst) to 5 (best):
    1. faithfulness: does every claim in the answer come from the context, with no invented facts?
    2. relevancy: does the answer directly address the question?

    Respond with only a JSON object in this exact format, nothing else:
    {{"faithfulness": <int>, "relevancy": <int>}}
    """
    judgment = llm.invoke(judge_prompt).content
    return parse_llm_json(judgment)


def evaluate_generation(eval_set, client, tokenizer, model, n=10):
    scores = []
    for item in eval_set[:n]:
        answer, sources = generate_answer(item["question"], client, tokenizer, model)
        context = "\n\n".join(s["content"] for s in sources)
        scores.append(judge_answer(item["question"], context, answer))

    return {
        "faithfulness": sum(s["faithfulness"] for s in scores) / len(scores),
        "relevancy": sum(s["relevancy"] for s in scores) / len(scores),
    }


# ---- Retrieval robustness: chunk-size / index-size grid search ----

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def max_similarity_to_targets(embedding, target_embeddings):
    return max(cosine_similarity(embedding, target) for target in target_embeddings)


def run_robustness_grid(chunk_sizes, distractor_checkpoints, target_documents, distractor_pool, eval_set, tokenizer, model, k=5):
    results = []

    for chunk_size in chunk_sizes:
        for checkpoint in distractor_checkpoints:
            trial_documents = target_documents + distractor_pool[:checkpoint]
            trial_splits = recursive_split_documents(trial_documents, chunk_size=chunk_size)
            trial_embeddings = [get_text_embeddings(doc.page_content, tokenizer, model) for doc in trial_splits]

            trial_client = QdrantClient(":memory:")
            index_documents(trial_client, "research_collection", trial_splits, trial_embeddings)

            scores = evaluate_retrieval(eval_set, trial_client, tokenizer, model, k=k)

            results.append({
                "chunk_size": chunk_size,
                "distractor_count": checkpoint,
                "indexed_chunks": len(trial_splits),
                "recall@5": scores["recall@k"],
                "mrr": scores["mrr"],
            })

            print(
                f"chunk_size={chunk_size:>5} | +{checkpoint:>3} distractor docs | "
                f"{len(trial_splits):>4} chunks indexed | "
                f"recall@5={scores['recall@k']:.2f} | mrr={scores['mrr']:.2f}"
            )

    return pd.DataFrame(results)


def summarize_robustness(results_df, distractor_checkpoints, label=""):
    from IPython.display import display

    pivot_recall = results_df.pivot(index="chunk_size", columns="distractor_count", values="recall@5")
    pivot_mrr = results_df.pivot(index="chunk_size", columns="distractor_count", values="mrr")

    print(f"Recall@5 by chunk_size (rows) x distractor documents added (columns){label}:")
    display(pivot_recall)

    print(f"\nMRR by chunk_size (rows) x distractor documents added (columns){label}:")
    display(pivot_mrr)

    smallest, largest = min(distractor_checkpoints), max(distractor_checkpoints)
    recall_drop = (pivot_recall[smallest] - pivot_recall[largest]).rename("recall_drop")

    summary = pd.concat(
        [pivot_recall[largest].rename("recall@5_at_largest_index"), recall_drop],
        axis=1,
    )

    print(f"\nStability summary (lower recall_drop = more robust to a growing index){label}:")
    display(summary)

    most_robust = summary["recall_drop"].idxmin()
    best_at_scale = summary["recall@5_at_largest_index"].idxmax()
    print(f"\nMost robust to index growth: chunk_size={most_robust} (smallest recall_drop)")
    print(f"Best absolute recall at the largest index tested: chunk_size={best_at_scale}")

    return pivot_recall, pivot_mrr, summary


def plot_recall_decay(results_df, chunk_sizes, title):
    import matplotlib.pyplot as plt

    series_colors = {512: "#2a78d6", 1024: "#eb6834", 2048: "#1baf7a"}

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    for chunk_size in chunk_sizes:
        series = results_df[results_df["chunk_size"] == chunk_size].sort_values("distractor_count")
        color = series_colors.get(chunk_size, "#4a3aa7")
        ax.plot(
            series["distractor_count"],
            series["recall@5"],
            color=color,
            linewidth=2,
            marker="o",
            markersize=8,
            label=f"chunk_size={chunk_size}",
        )
        last = series.iloc[-1]
        ax.annotate(
            f"  {chunk_size}",
            (last["distractor_count"], last["recall@5"]),
            color=color,
            fontsize=9,
            va="center",
        )

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Distractor documents added to index", color="#52514e")
    ax.set_ylabel("Recall@5", color="#52514e")
    ax.set_title(title, color="#0b0b0b", fontsize=13, pad=12)

    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c3c2b7")
    ax.tick_params(colors="#898781")

    ax.legend(frameon=False, loc="lower left")

    plt.tight_layout()
    plt.show()


# ---- Hotel-review pipeline: FAISS ----

def get_embeddings(data, model):
    embeddings = model.encode(data, show_progress_bar=False).astype("float32")
    return embeddings


def create_faiss_index(embeddings):
    embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(embeddings_normalized.shape[1])
    index.add(embeddings_normalized)
    return index


def search_faiss_index(query_embedding, faiss_index, k=5):
    query_embedding_normalized = query_embedding / np.linalg.norm(query_embedding)
    distances, indices = faiss_index.search(query_embedding_normalized, k)
    return distances, indices


def search_hotels_by_query(query, model, faiss_index, df, k=25):
    query_embedding = get_embeddings([query], model)
    distances, indices = search_faiss_index(query_embedding, faiss_index, k=k)

    results = []
    for i, (idx, distance) in enumerate(zip(indices[0], distances[0]), 1):
        results.append({
            "rank": i,
            "hotel_name": df.iloc[idx]["hotel_name"],
            "review_text": df.iloc[idx]["review_text"],
            "cosine_similarity": float(distance),
        })

    return results


def generate_hotel_answer_faiss(query, model, faiss_index, df):
    json_output = search_hotels_by_query(query, model, faiss_index, df, k=25)
    prompt = f"""
    Based on the following query from a user, please generate a small answer
    focusing on the original query and the response given. The answer should be paragraphs.
    Remove the special characters and (/n), make the output clean and long.
    Please cite source for each part as [1][2].
    Just start with the answer, no need to give any salutations.

    ###########
    query:
    "{query}"

    ########

    context:
    "{json_output}"
    #####

    Return in Markdown format.
    """
    output_text = ""
    with anthropic_client.messages.stream(
        model="claude-haiku-4-5",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            output_text += text
            print(text, end="")

    return output_text, json_output


# ---- Hotel-review pipeline: Qdrant ----

def create_qdrant_collection(client, collection_name, vector_size):
    try:
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name=collection_name)
            print(f"Collection '{collection_name}' deleted successfully.")

        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

        print(f"Collection '{collection_name}' created successfully.")

    except Exception as e:
        print(f"An error occurred while setting up the collection: {e}")


def upload_reviews_to_qdrant(client, collection_name, reviews_df, review_embeddings):
    points_to_upload = [
        models.PointStruct(
            id=i,
            vector=review_embeddings[i].tolist(),
            payload={
                "hotel_name": reviews_df.iloc[i]["hotel_name"],
                "review_text": reviews_df.iloc[i]["review_text"],
                "locality": reviews_df.iloc[i]["locality"],
            },
        )
        for i in range(len(review_embeddings))
    ]

    return client.upsert(
        collection_name=collection_name,
        wait=True,
        points=points_to_upload,
    )


def search_qdrant_with_filter(query, model, client, city=None, k=10):
    query_embedding = get_embeddings([query], model)[0]

    query_filter = None
    if city:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="locality",
                    match=models.MatchValue(value=city),
                )
            ]
        )

    text_hits = client.query_points(
        collection_name="hotel_reviews",
        query=query_embedding,
        query_filter=query_filter,
        limit=k,
        with_payload=True,
    ).points

    return text_hits


def generate_hotel_answer_qdrant(query, model, client, city=None):
    qdrant_results = search_qdrant_with_filter(query, model, client, city=city, k=25)

    context_string = ""
    for i, result in enumerate(qdrant_results):
        context_string += f"Source {i+1}:\n"
        context_string += f"Hotel: {result.payload.get('hotel_name', 'N/A')}\n"
        context_string += f"Review: {result.payload.get('review_text', 'N/A')}\n"
        context_string += f"Locality: {result.payload.get('locality', 'N/A')}\n"
        context_string += f"Similarity Score: {result.score:.4f}\n\n"

    prompt = f"""
    Based on the following query from a user and the provided context from hotel reviews,
    please generate a concise answer summarizing the relevant information.
    Focus on addressing the user's query using details found in the reviews.
    Cite the sources using numerical references like [1], [2], etc., corresponding to the "Source #" in the context.
    Format the output as a few paragraphs.
    Remove any special characters like (/n) and ensure the output is clean.
    Begin directly with the answer.

    ###########
    query:
    "{query}"

    ########

    context:
    "{context_string}"
    #####

    Return in Markdown format.
    """

    output_text = ""
    with anthropic_client.messages.stream(
        model="claude-haiku-4-5",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            output_text += text
            print(text, end="")

    return output_text, qdrant_results
