import io
from pathlib import Path
from typing import List

import requests


## TODO: Make this token-aware.
def convert_messages_to_str(messages: List[dict]) -> str:
    """convert messages in OpenAI API format into a single string.

    Args:
        messages List[dict]: A list of dictionary in the following format:
        ```
        [
          {"role": "user", "content": "Hello there."},
          {"role": "assistant", "content": "Hi, I'm Claude. How can I help you?"},
          {"role": "user", "content": "Can you explain LLMs in plain English?"},
        ]
        ```
    """
    final_str = ""
    for msg in messages:
        if "role" in msg and "content" in msg:
            role = msg.get("role", "")
            content = msg.get("content", "")
            final_str += f"\n\n{role}: {content}"
    return final_str


def format_assistant_message(raw_text: str) -> dict:
    return {"role": "assistant", "content": raw_text}


def load_documents(file_path: str = None, folder_path: str = None, url: str = None) -> List[any]:
    """Loads documents from a local directory or public url or local filename.

    Args:
        file_path (str): The path to the filename.
        folder_path (str): The path to the folder.
        url (str): The url to the file.
    """
    # check import packages
    try:
        from llama_index.core import Document, SimpleDirectoryReader
        from llama_index.core.readers.download import download_loader
    except ImportError:
        raise ImportError(
            "Could not import llama index package. "
            "Please install it with `pip install llama-index-core==0.10.1`."
        )
    # document loaders for filepath
    if file_path:
        if file_path.endswith(".pdf"):
            PDFReader = download_loader("PDFReader")
            loader = PDFReader()
            documents = loader.load_data(file=Path(file_path))
        elif file_path.endswith(".docx"):
            docReader = download_loader("DocxReader")
            loader = docReader()
            documents = loader.load_data(file=Path(file_path))
        elif file_path.endswith(".txt"):
            with open(file_path, 'r') as file:
                text_content = file.read()
            documents = [Document(text=text_content)]
        else:
            raise ValueError("Only .pdf, .docx, and .txt files are supported.")

    # document loaders for folderpath
    if folder_path:
        documents = SimpleDirectoryReader(
            input_dir=Path(folder_path), required_exts=[".pdf", ".docx", ".txt"]
        ).load_data()

    # document loaders for url
    if url:
        response = requests.get(url)
        if response.status_code != 200:
            raise ValueError(f"Invalid url {url}.")
        # for text files
        try:
            documents = [Document(text=response.content)]
        # for pdf files
        except Exception:
            # check import packages
            try:
                from pypdf import PdfReader
            except ImportError:
                raise ImportError(
                    "Could not import pypdf package. "
                    "Please install it with `pip install pypdf==3.17.4`."
                )
            documents = []
            pdf_file = PdfReader(io.BytesIO(response.content))
            num_pages = len(pdf_file.pages)
            for page in range(num_pages):
                page_text = pdf_file.pages[page].extract_text()
                documents.append(Document(text=page_text))
        else:
            raise ValueError(f"Invalid url {url}.")

    return documents


def split_document(text: str, chunk_size: int, chunk_overlap: int, **kwargs) -> List[str]:
    """Splits a document into chunks of text.

    Args:
        text (str): The text to split.
        chunk_size (int): The size of each chunk.
        chunk_overlap (int): The amount of overlap between each chunk.
        **kwargs: Additional keyword arguments for the SentenceSplitter.
    """
    # check import packages
    try:
        from llama_index.core.node_parser.text import SentenceSplitter
    except ImportError:
        raise ImportError(
            "Could not import llama index package. "
            "Please install it with `pip install llama-index-core==0.10.24`."
        )
    # document
    text_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
    text_chunks = text_parser.split_text(text)
    return text_chunks
