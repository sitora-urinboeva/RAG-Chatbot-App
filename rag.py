from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
import os

class Embeddings():
    def __init__(self, model_name, device, qdrant_url, collection_name, encode_kwargs):
        self.model_name = model_name
        self.device = device
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.encode_kwargs = encode_kwargs
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs=self.device,
            encode_kwargs=self.encode_kwargs)

    def create_embeddings(self, pdf_path):
        if os.path.exists(pdf_path) is None:
            raise FileNotFoundError('THere is no file')
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        splits = text_splitter.split_documents(docs)
        Qdrant.from_documents(
            splits,
            self.embeddings,
            url=self.qdrant_url,
            collection_name=self.collection_name
        )
        return 'Vector Base is created!'
