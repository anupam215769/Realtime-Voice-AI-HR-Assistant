import os
import re
import shutil
from dotenv import load_dotenv
from llama_index.core.node_parser import SentenceSplitter
from langchain_openai import AzureOpenAIEmbeddings
from llama_cloud_services import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import chromadb
load_dotenv()

embedding = AzureOpenAIEmbeddings(
    api_key=os.getenv("AZURE_OPENAI_CODE_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_CODE_ENDPOINT"),
    azure_deployment="text-embedding-3-large", 
    api_version="2024-12-01-preview",
    chunk_size=2000
)


parser = LlamaParse(api_key=os.getenv("LLAMA_PARSE_KEY"), result_type="markdown")
file_extractor = {".pdf": parser}
results = SimpleDirectoryReader("./pdfs", file_extractor=file_extractor).load_data()


## Creating chunks of data
print("Chunking")
text_splitter = SentenceSplitter(chunk_size=2000, chunk_overlap=200)
documents = text_splitter.get_nodes_from_documents(results)

print("Creating embeddings and storing in docdb")
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("pdfs")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embedding
)