"""
Document processing module for loading and splitting documents.
"""

import os
from pathlib import Path
from typing import List, Union

from dotenv import load_dotenv
load_dotenv()

os.environ["USER_AGENT"] = os.getenv("USER_AGENT")

from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
    PyPDFDirectoryLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DocumentProcessor:
    """Handles document loading and splitting into manageable text chunks."""

    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 50):
        """
        Args:
            chunk_size: Maximum size of chunks after splitting.
            chunk_overlap: Number of overlapping characters between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    # ----------------------------------------------------------------------
    # Loading functions
    # ----------------------------------------------------------------------

    def load_from_url(self, url: str) -> List[Document]:
        """Load documents from a webpage URL."""
        loader = WebBaseLoader(url)
        return loader.load()

    def load_from_pdf(self, file_path: Union[str, Path]) -> List[Document]:
        """Load a single PDF file."""
        loader = PyPDFLoader(str(file_path))
        return loader.load()

    def load_from_pdf_dir(self, directory: Union[str, Path]) -> List[Document]:
        """Load all PDF files inside a directory."""
        loader = PyPDFDirectoryLoader(str(directory))
        return loader.load()

    def load_from_txt(self, file_path: Union[str, Path]) -> List[Document]:
        """Load the text contents of a TXT file as a single Document."""
        loader = TextLoader(str(file_path), encoding="utf-8")
        return loader.load()

    def load_urls_from_txt(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Load URLs stored line-by-line in a TXT file and fetch each webpage.
        """
        urls = [
            line.strip() for line in Path(file_path).read_text().splitlines()
            if line.strip()
        ]

        docs: List[Document] = []
        for url in urls:
            docs.extend(self.load_from_url(url))

        return docs

    # ----------------------------------------------------------------------
    # General multipurpose loader
    # ----------------------------------------------------------------------

    def load_sources(self, sources: List[str]) -> List[Document]:
        """
        Load documents from URLs, PDF files/directories, or .txt files.

        Args:
            sources: List of paths/URLs.

        Returns:
            Loaded documents.
        """
        docs: List[Document] = []

        for src in sources:
            path = Path(src)

            if src.startswith(("http://", "https://")):
                docs.extend(self.load_from_url(src))

            elif path.is_dir():
                docs.extend(self.load_from_pdf_dir(path))

            elif path.suffix.lower() == ".pdf":
                docs.extend(self.load_from_pdf(path))

            elif path.suffix.lower() == ".txt":
                # If the txt file contains URLs, load them:
                if self._txt_contains_urls(path):
                    docs.extend(self.load_urls_from_txt(path))
                else:
                    docs.extend(self.load_from_txt(path))

            else:
                raise ValueError(f"Unsupported source type: {src}")

        return docs

    def _txt_contains_urls(self, file_path: Path) -> bool:
        """Check whether a TXT file contains URLs."""
        for line in file_path.read_text().splitlines():
            if line.strip().startswith(("http://", "https://")):
                return True
        return False

    # ----------------------------------------------------------------------
    # Splitting
    # ----------------------------------------------------------------------

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split loaded documents into chunks."""
        return self.splitter.split_documents(documents)

    def process(self, sources: List[str]) -> List[Document]:
        """
        Complete pipeline: load -> split.

        Args:
            sources: List of file paths or URLs.

        Returns:
            Chunked documents.
        """
        docs = self.load_sources(sources)
        return self.split_documents(docs)
