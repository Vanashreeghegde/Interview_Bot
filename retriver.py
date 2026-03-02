import os
from langchain_chroma import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

class HybridRetriever:
    def __init__(self, persist_directory="Database"):

        # Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Load Chroma DB
        self.vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_name="ds_interview_bot"
        )

        # Dense retriever
        self.dense_retriever = self.vectordb.as_retriever(
            search_kwargs={"k": 4}
        )

        # Load stored documents for BM25
        raw_data = self.vectordb.get()
        num_docs = len(raw_data["documents"])
        print("Documents in DB:", num_docs)

        if num_docs == 0:
            print("⚠️ Database is empty. Run ingestion first.")
            self.sparse_retriever = None
            return

        docs = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(raw_data["documents"], raw_data["metadatas"])
        ]

        # Sparse retriever
        self.sparse_retriever = BM25Retriever.from_documents(docs)
        self.sparse_retriever.k = 4

    def retrieve(self, query: str):
        dense_docs = self.dense_retriever.invoke(query)
        sparse_docs = self.sparse_retriever.invoke(query)
        combined = dense_docs + sparse_docs

        # Remove duplicates
        seen = set()
        unique_docs = []
        for doc in combined:
            if doc.page_content not in seen:
                unique_docs.append(doc)
                seen.add(doc.page_content)

        return unique_docs[:4]