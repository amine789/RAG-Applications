import sys

sys.path.append("..")
from llm_model import llm


def split_and_import(loader, text_splitter, vector_db):
    chunks = text_splitter.split_documents(loader.load())
    vector_db.add_documents(chunks)
    print(f"Ingested chunks created by {loader}")


def execute_chain(chain, question):
    answer = chain.invoke(question)
    return answer



