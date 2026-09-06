import sys
from pathlib import Path

import chromadb
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.append("..")
from llm_model import llm


def get_tourism_collection():
    chroma_client = chromadb.Client()
    return chroma_client.get_or_create_collection(name="tourism_coolection")


tourism_collection = get_tourism_collection()


def load_and_split_documents(chunk_size=500, chunk_overlap=50):
    word_docs = Docx2txtLoader("../Paestrum/Paestum-Britannica.docx").load()
    #pdf_docs = PyPDFLoader("../Paestrum/PaestumRevisited.pdf").load()
    pdf_docs_stockholm = PyPDFLoader("../Paestrum/PaestumRevisited-StocholmsUniversitet.pdf").load()
    txt_docs = TextLoader("../Paestrum/Paestum-Encyclopedia.txt").load()

    all_docs = word_docs + pdf_docs_stockholm + txt_docs

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(all_docs)


def add_documents_to_collection(split_docs):
    tourism_collection.add(
        documents=[doc.page_content for doc in split_docs],
        metadatas=[doc.metadata for doc in split_docs],
        ids=[f"{Path(doc.metadata['source']).stem}-{i}" for i, doc in enumerate(split_docs)]
    )


def query_vector_db(question):
    results = tourism_collection.query(
        query_texts=[question],
        n_results=3
    )
    results_text = results
    return results_text


def chatbot(question):
    context = query_vector_db(question)

    prompt = f"""Answer the question using only the context below.
If the context doesn't contain the answer, say so.

Context:
{context}

Question: {question}"""

    response = llm.invoke(prompt)
    return response.content
