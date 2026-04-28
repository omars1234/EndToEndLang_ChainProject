from langchain_community.document_loaders import TextLoader,PyPDFLoader,PyMuPDFLoader,CSVLoader,WebBaseLoader,DirectoryLoader,YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter,CharacterTextSplitter,TokenTextSplitter#SemanticChunker
from langchain_core.documents import Document
from pathlib import Path
from langchain_community.vectorstores import FAISS,Chroma

from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_groq import ChatGroq
from langchain.chat_models import init_chat_model
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface  import HuggingFaceEmbeddings





import os
from dotenv import load_dotenv
load_dotenv()
load_dotenv(dotenv_path=os.path.join(os.path.dirname("../"), ".env"))


os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

llm = init_chat_model("groq:openai/gpt-oss-120b")


def web_loader_fuction(web_path):
    print("Loading web content...")
    web_loader=WebBaseLoader([web_path],
                             headers={"User-Agent": os.getenv("USER_AGENT")})
    documents=web_loader.load()
    print(f"Loaded {len(documents)} documents from the web.")
    return documents


def PDF_loader_fuction(pdf_file):
    print("Loading PDF content...")
    pdf_loader=PyMuPDFLoader(pdf_file)
    documents=pdf_loader.load()
    print(f"Loaded {len(documents)} documents from the PDF.")
    return documents


def spiltter_function(documents):
    print("Splitting documents into chunks...")
    text_splitter=RecursiveCharacterTextSplitter(separators="\n",chunk_size=1000,chunk_overlap=200)
    chunks=text_splitter.split_documents(documents=documents) 
    print(f"Split into {len(chunks)} chunks.")
    return chunks


def embedding_function():    
    print("Creating embeddings...")
    embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("Embedding created.")
    return   embedding



def vector_store_function(documents,embedding):
    print("Creating vector store...")
    vector_store=FAISS.from_documents(documents=documents,embedding=embedding)
    print("Vector store created.")
    return vector_store



def Create_documents_chain(llm,prompt,retriever):
    print("Creating documents chain...")
    documents_chain=create_stuff_documents_chain(llm,prompt)
    print("Documents chain created.")
    print("Creating retrieval chain...")
    retriever_chain=create_retrieval_chain(retriever,documents_chain)
    print("Retrieval chain created.")
    return retriever_chain
   

