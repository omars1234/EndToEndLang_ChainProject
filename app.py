
import os

#from turtle import pdf

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
from EndToEndLangChainProject.prompt import system_prompt
from langchain_huggingface  import HuggingFaceEmbeddings

from EndToEndLangChainProject.helper import PDF_loader_fuction, web_loader_fuction,spiltter_function,embedding_function,vector_store_function,Create_documents_chain
import time
import tempfile

import os
from dotenv import load_dotenv
load_dotenv()
load_dotenv(dotenv_path=os.path.join(os.path.dirname("../"), ".env"))


os.environ["USER_AGENT"] = os.getenv("USER_AGENT")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

llm = init_chat_model("groq:openai/gpt-oss-120b")

import streamlit as st



st.title("Rag-langchain demo with groq-oss-120b")

option = st.selectbox(
    "what is the documents formtat you want to use ?",
    ("PDF", "HTML Web")
)

if option == "PDF":
    pdf_file = st.file_uploader("Upload your PDF file here", type=["pdf"])
    load_button = st.button("Load and Process PDF")

    if pdf_file is not None and load_button:
        try:
            with st.spinner("Processing..."):
                # Read ONCE
                file_bytes = pdf_file.read()

                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(file_bytes)
                    temp_file_path = tmp_file.name

                # Process
                pdf_loader = PDF_loader_fuction(temp_file_path)
                splitter = spiltter_function(pdf_loader)
                embedding = embedding_function()
                chroma_db = vector_store_function(splitter, embedding)

                # Clean up
                os.remove(temp_file_path)

            st.session_state.Chroma_db = chroma_db
            st.success("✅ PDF processed!")

        except Exception as e:
            st.error(f"Error: {str(e)}")

elif option == "HTML Web":            

    web_url = st.text_input("Input your URL here")
    load_button = st.button("Load and Process URL")

    # Step 1: Create DB
    if web_url and load_button:
        try:
            with st.spinner("Processing..."):
                web_loader = web_loader_fuction(web_url)
                splitter = spiltter_function(web_loader)
                embedding = embedding_function()
                chroma_db = vector_store_function(splitter, embedding)

            st.session_state.Chroma_db = chroma_db
            st.success("✅ URL processed!")

        except Exception as e:
            st.error(f"Error: {str(e)}")



prompt=ChatPromptTemplate.from_messages(
        [
        ("system", system_prompt),
        ("human", "{input}")
        ]
    )

# Step 2: Use DB (ONLY if exists)
if "Chroma_db" in st.session_state:
     
    retriever=st.session_state.Chroma_db.as_retriever()
    documents_chain=create_stuff_documents_chain(llm,prompt)
    retriever_chain=create_retrieval_chain(retriever,documents_chain)
        

    prompt=st.text_input("input your prompt here")
    if  prompt:
            start=time.process_time()
            response=retriever_chain.invoke({"input":prompt})
            print("Response time",time.process_time()-start)
            st.write(response["answer"])
            st.download_button("Download response TXT",response["answer"],file_name="response.txt")
            st.download_button("Download response DOCX",response["answer"],file_name="response.docx")
            st.download_button("Download response PDF",response["answer"],file_name="response.pdf")

            

            with st.expander("Documnets similariry search"):
                for i ,doc in enumerate(response["context"]):
                    st.write(doc.page_content)
                    st.write("---------------------")


def reset_all():
    for key in st.session_state.keys():
        del st.session_state[key]

st.button("Reset Everything", on_click=reset_all)
