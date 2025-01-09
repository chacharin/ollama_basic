from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import sys

def ingest():
    # Get the doc
    loader = PyPDFLoader("data\history_of_computer.pdf")
    pages = loader.load_and_split()
    # Split the pages by char
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(pages)
    print(f"Split {len(pages)} documents into {len(chunks)} chunks.")
    #
    embedding = FastEmbedEmbeddings()
    #Create vector store
    Chroma.from_documents(documents=chunks,  embedding=embedding, persist_directory="./knowledge")
    
# only run this once to generate vector store
ingest()