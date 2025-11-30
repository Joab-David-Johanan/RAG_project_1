"""
Vectorstore creation module for document embedding and retrieval.
"""

from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class VectorStoreManager:
    """
    Manages vector store creation, retrieval, and persistence.
    """

    def __init__(self, embedding_model=None):
        """
        Initialize the vectorstore manager.

        Args:
            embedding_model: Optional custom embedding model.
        """
        self.embedding = embedding_model or OpenAIEmbeddings()
        self.vectorstore: Optional[FAISS] = None
        self.retriever: Optional[BaseRetriever] = None

    # ----------------------------------------------------------------------
    # Creation
    # ----------------------------------------------------------------------

    def create_vectorstore(self, documents: List[Document]):
        """
        Create a FAISS vectorstore from given documents.

        Args:
            documents: List of documents to embed.

        Returns:
            self (to enable method chaining)
        """
        # Filter empty documents
        documents = [d for d in documents if d.page_content.strip()]

        if not documents:
            raise ValueError("No valid documents provided to create vectorstore.")

        self.vectorstore = FAISS.from_documents(documents, self.embedding)
        self.retriever = self.vectorstore.as_retriever()
        return self

    # ----------------------------------------------------------------------
    # Retrieval
    # ----------------------------------------------------------------------

    def get_retriever(self) -> BaseRetriever:
        """Return the retriever instance."""
        if self.retriever is None:
            raise ValueError("Vectorstore not initialized. Call create_vectorstore().")
        return self.retriever

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Query text.
            k: Number of documents to retrieve.

        Returns:
            List of top-k matching documents.
        """
        if self.retriever is None:
            raise ValueError("Vectorstore not initialized. Call create_vectorstore().")

        return self.retriever.get_relevant_documents(query)

    # ----------------------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------------------

    def save(self, folder: str):
        """Save vectorstore to local directory."""
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized.")
        self.vectorstore.save_local(folder)

    def load(self, folder: str):
        """Load vectorstore from local directory."""
        self.vectorstore = FAISS.load_local(folder, self.embedding)
        self.retriever = self.vectorstore.as_retriever()
