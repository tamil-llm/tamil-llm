"""
MIT License

Author: Chandra Sakthivel
Date: 2024-06-02

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
"""

import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.document_loaders.base import Document

CACHE_DIR = "./LLM_RAG_Bot/models"

class FaissDb:
    def __init__(self, docs, embedding_function):
        self.embedding_function = embedding_function
        texts = [doc.page_content if hasattr(doc, 'page_content') else doc for doc in docs]
        embeddings = self.embedding_function.encode(texts)
        text_embeddings = list(zip(texts, embeddings))
        self.db = FAISS.from_embeddings(
            text_embeddings=text_embeddings,
            embedding=self.embedding_function.encode,
            distance_strategy=DistanceStrategy.COSINE
        )

    def add_documents(self, docs):
        texts = [doc.page_content if hasattr(doc, 'page_content') else doc for doc in docs]
        embeddings = self.embedding_function.encode(texts)
        text_embeddings = list(zip(texts, embeddings))
        self.db.add_embeddings(text_embeddings)

    def size(self):
        return self.db.index.ntotal

    def similarity_search(self, question: str, k: int = 3):
        retrieved_docs = self.db.similarity_search(question, k=k)
        context = "".join(f"Page {doc.metadata.get('page', 'Unknown')}: {doc.page_content}\n" for doc in retrieved_docs)
        return context

def load_and_split_documents(file_paths: list, chunk_size: int = 256):
    loaders = []
    for file_path in file_paths:
        if file_path.endswith('.pdf'):
            loaders.append(PyPDFLoader(file_path))
        else:
            loaders.append(UnstructuredFileLoader(file_path, encoding='utf-8'))
    
    pages = []
    for loader in loaders:
        try:
            pages.extend(loader.load())
        except UnicodeDecodeError:
            with open(loader.file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
            pages.append(Document(page_content=content))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        strip_whitespace=True
    )
        
    docs = text_splitter.split_documents(pages)
    return docs

def add_new_documents_to_db(docs, db):
    existing_docs = set([doc.page_content for doc in db.db.docstore._dict.values()])
    new_docs = [doc for doc in docs if doc.page_content not in existing_docs]
    if new_docs:
        db.add_documents(new_docs)
    return new_docs
